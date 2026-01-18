from __future__ import annotations
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import Numeric
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import NULLTYPE
from .base import alter_column
from .base import alter_table
from .base import AlterColumn
from .base import ColumnComment
from .base import format_column_name
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import ComparisonResult
from .impl import DefaultImpl
from .. import util
from ..autogenerate import render
from ..operations import ops
from ..operations import schemaobj
from ..operations.base import BatchOperations
from ..operations.base import Operations
from ..util import sqla_compat
from ..util.sqla_compat import compiles
def autogen_column_reflect(self, inspector, table, column_info):
    if column_info.get('default') and isinstance(column_info['type'], (INTEGER, BIGINT)):
        seq_match = re.match("nextval\\('(.+?)'::regclass\\)", column_info['default'])
        if seq_match:
            info = sqla_compat._exec_on_inspector(inspector, text("select c.relname, a.attname from pg_class as c join pg_depend d on d.objid=c.oid and d.classid='pg_class'::regclass and d.refclassid='pg_class'::regclass join pg_class t on t.oid=d.refobjid join pg_attribute a on a.attrelid=t.oid and a.attnum=d.refobjsubid where c.relkind='S' and c.relname=:seqname"), seqname=seq_match.group(1)).first()
            if info:
                seqname, colname = info
                if colname == column_info['name']:
                    log.info("Detected sequence named '%s' as owned by integer column '%s(%s)', assuming SERIAL and omitting", seqname, table.name, colname)
                    del column_info['default']