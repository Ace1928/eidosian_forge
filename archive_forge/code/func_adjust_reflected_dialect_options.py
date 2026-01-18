from __future__ import annotations
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import types as sqltypes
from sqlalchemy.schema import Column
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.base import Executable
from sqlalchemy.sql.elements import ClauseElement
from .base import AddColumn
from .base import alter_column
from .base import alter_table
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .base import format_table_name
from .base import format_type
from .base import RenameTable
from .impl import DefaultImpl
from .. import util
from ..util import sqla_compat
from ..util.sqla_compat import compiles
def adjust_reflected_dialect_options(self, reflected_object: Dict[str, Any], kind: str) -> Dict[str, Any]:
    options: Dict[str, Any]
    options = reflected_object.get('dialect_options', {}).copy()
    if not options.get('mssql_include'):
        options.pop('mssql_include', None)
    if not options.get('mssql_clustered'):
        options.pop('mssql_clustered', None)
    return options