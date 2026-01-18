from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
@reflection.cache
def get_foreign_keys(self, connection, table_name, schema=None, **kw):
    pragma_fks = self._get_table_pragma(connection, 'foreign_key_list', table_name, schema=schema)
    fks = {}
    for row in pragma_fks:
        numerical_id, rtbl, lcol, rcol = (row[0], row[2], row[3], row[4])
        if not rcol:
            try:
                referred_pk = self.get_pk_constraint(connection, rtbl, schema=schema, **kw)
                referred_columns = referred_pk['constrained_columns']
            except exc.NoSuchTableError:
                referred_columns = []
        else:
            referred_columns = []
        if self._broken_fk_pragma_quotes:
            rtbl = re.sub('^[\\"\\[`\\\']|[\\"\\]`\\\']$', '', rtbl)
        if numerical_id in fks:
            fk = fks[numerical_id]
        else:
            fk = fks[numerical_id] = {'name': None, 'constrained_columns': [], 'referred_schema': schema, 'referred_table': rtbl, 'referred_columns': referred_columns, 'options': {}}
            fks[numerical_id] = fk
        fk['constrained_columns'].append(lcol)
        if rcol:
            fk['referred_columns'].append(rcol)

    def fk_sig(constrained_columns, referred_table, referred_columns):
        return tuple(constrained_columns) + (referred_table,) + tuple(referred_columns)
    keys_by_signature = {fk_sig(fk['constrained_columns'], fk['referred_table'], fk['referred_columns']): fk for fk in fks.values()}
    table_data = self._get_table_sql(connection, table_name, schema=schema)

    def parse_fks():
        if table_data is None:
            return
        FK_PATTERN = '(?:CONSTRAINT (\\w+) +)?FOREIGN KEY *\\( *(.+?) *\\) +REFERENCES +(?:(?:"(.+?)")|([a-z0-9_]+)) *\\( *((?:(?:"[^"]+"|[a-z0-9_]+) *(?:, *)?)+)\\) *((?:ON (?:DELETE|UPDATE) (?:SET NULL|SET DEFAULT|CASCADE|RESTRICT|NO ACTION) *)*)((?:NOT +)?DEFERRABLE)?(?: +INITIALLY +(DEFERRED|IMMEDIATE))?'
        for match in re.finditer(FK_PATTERN, table_data, re.I):
            constraint_name, constrained_columns, referred_quoted_name, referred_name, referred_columns, onupdatedelete, deferrable, initially = match.group(1, 2, 3, 4, 5, 6, 7, 8)
            constrained_columns = list(self._find_cols_in_sig(constrained_columns))
            if not referred_columns:
                referred_columns = constrained_columns
            else:
                referred_columns = list(self._find_cols_in_sig(referred_columns))
            referred_name = referred_quoted_name or referred_name
            options = {}
            for token in re.split(' *\\bON\\b *', onupdatedelete.upper()):
                if token.startswith('DELETE'):
                    ondelete = token[6:].strip()
                    if ondelete and ondelete != 'NO ACTION':
                        options['ondelete'] = ondelete
                elif token.startswith('UPDATE'):
                    onupdate = token[6:].strip()
                    if onupdate and onupdate != 'NO ACTION':
                        options['onupdate'] = onupdate
            if deferrable:
                options['deferrable'] = 'NOT' not in deferrable.upper()
            if initially:
                options['initially'] = initially.upper()
            yield (constraint_name, constrained_columns, referred_name, referred_columns, options)
    fkeys = []
    for constraint_name, constrained_columns, referred_name, referred_columns, options in parse_fks():
        sig = fk_sig(constrained_columns, referred_name, referred_columns)
        if sig not in keys_by_signature:
            util.warn("WARNING: SQL-parsed foreign key constraint '%s' could not be located in PRAGMA foreign_keys for table %s" % (sig, table_name))
            continue
        key = keys_by_signature.pop(sig)
        key['name'] = constraint_name
        key['options'] = options
        fkeys.append(key)
    fkeys.extend(keys_by_signature.values())
    if fkeys:
        return fkeys
    else:
        return ReflectionDefaults.foreign_keys()