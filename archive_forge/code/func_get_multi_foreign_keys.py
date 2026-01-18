from the proposed insertion.   These values are specified using the
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import arraylib as _array
from . import json as _json
from . import pg_catalog
from . import ranges as _ranges
from .ext import _regconfig_fn
from .ext import aggregate_order_by
from .hstore import HSTORE
from .named_types import CreateDomainType as CreateDomainType  # noqa: F401
from .named_types import CreateEnumType as CreateEnumType  # noqa: F401
from .named_types import DOMAIN as DOMAIN  # noqa: F401
from .named_types import DropDomainType as DropDomainType  # noqa: F401
from .named_types import DropEnumType as DropEnumType  # noqa: F401
from .named_types import ENUM as ENUM  # noqa: F401
from .named_types import NamedType as NamedType  # noqa: F401
from .types import _DECIMAL_TYPES  # noqa: F401
from .types import _FLOAT_TYPES  # noqa: F401
from .types import _INT_TYPES  # noqa: F401
from .types import BIT as BIT
from .types import BYTEA as BYTEA
from .types import CIDR as CIDR
from .types import CITEXT as CITEXT
from .types import INET as INET
from .types import INTERVAL as INTERVAL
from .types import MACADDR as MACADDR
from .types import MACADDR8 as MACADDR8
from .types import MONEY as MONEY
from .types import OID as OID
from .types import PGBit as PGBit  # noqa: F401
from .types import PGCidr as PGCidr  # noqa: F401
from .types import PGInet as PGInet  # noqa: F401
from .types import PGInterval as PGInterval  # noqa: F401
from .types import PGMacAddr as PGMacAddr  # noqa: F401
from .types import PGMacAddr8 as PGMacAddr8  # noqa: F401
from .types import PGUuid as PGUuid
from .types import REGCLASS as REGCLASS
from .types import REGCONFIG as REGCONFIG  # noqa: F401
from .types import TIME as TIME
from .types import TIMESTAMP as TIMESTAMP
from .types import TSVECTOR as TSVECTOR
from ... import exc
from ... import schema
from ... import select
from ... import sql
from ... import util
from ...engine import characteristics
from ...engine import default
from ...engine import interfaces
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine import URL
from ...engine.reflection import ReflectionDefaults
from ...sql import bindparam
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.visitors import InternalTraversal
from ...types import BIGINT
from ...types import BOOLEAN
from ...types import CHAR
from ...types import DATE
from ...types import DOUBLE_PRECISION
from ...types import FLOAT
from ...types import INTEGER
from ...types import NUMERIC
from ...types import REAL
from ...types import SMALLINT
from ...types import TEXT
from ...types import UUID as UUID
from ...types import VARCHAR
from ...util.typing import TypedDict
def get_multi_foreign_keys(self, connection, schema, filter_names, scope, kind, postgresql_ignore_search_path=False, **kw):
    preparer = self.identifier_preparer
    has_filter_names, params = self._prepare_filter_names(filter_names)
    query = self._foreing_key_query(schema, has_filter_names, scope, kind)
    result = connection.execute(query, params)
    FK_REGEX = self._fk_regex_pattern
    fkeys = defaultdict(list)
    default = ReflectionDefaults.foreign_keys
    for table_name, conname, condef, conschema, comment in result:
        if conname is None:
            fkeys[schema, table_name] = default()
            continue
        table_fks = fkeys[schema, table_name]
        m = re.search(FK_REGEX, condef).groups()
        constrained_columns, referred_schema, referred_table, referred_columns, _, match, _, onupdate, _, ondelete, deferrable, _, initially = m
        if deferrable is not None:
            deferrable = True if deferrable == 'DEFERRABLE' else False
        constrained_columns = [preparer._unquote_identifier(x) for x in re.split('\\s*,\\s*', constrained_columns)]
        if postgresql_ignore_search_path:
            if conschema != self.default_schema_name:
                referred_schema = conschema
            else:
                referred_schema = schema
        elif referred_schema:
            referred_schema = preparer._unquote_identifier(referred_schema)
        elif schema is not None and schema == conschema:
            referred_schema = schema
        referred_table = preparer._unquote_identifier(referred_table)
        referred_columns = [preparer._unquote_identifier(x) for x in re.split('\\s*,\\s', referred_columns)]
        options = {k: v for k, v in [('onupdate', onupdate), ('ondelete', ondelete), ('initially', initially), ('deferrable', deferrable), ('match', match)] if v is not None and v != 'NO ACTION'}
        fkey_d = {'name': conname, 'constrained_columns': constrained_columns, 'referred_schema': referred_schema, 'referred_table': referred_table, 'referred_columns': referred_columns, 'options': options, 'comment': comment}
        table_fks.append(fkey_d)
    return fkeys.items()