from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
def pre_exec(self):
    """Activate IDENTITY_INSERT if needed."""
    if self.isinsert:
        if TYPE_CHECKING:
            assert is_sql_compiler(self.compiled)
            assert isinstance(self.compiled.compile_state, DMLState)
            assert isinstance(self.compiled.compile_state.dml_table, TableClause)
        tbl = self.compiled.compile_state.dml_table
        id_column = tbl._autoincrement_column
        if id_column is not None and (not isinstance(id_column.default, Sequence)):
            insert_has_identity = True
            compile_state = self.compiled.dml_compile_state
            self._enable_identity_insert = id_column.key in self.compiled_parameters[0] or (compile_state._dict_parameters and id_column.key in compile_state._insert_col_keys)
        else:
            insert_has_identity = False
            self._enable_identity_insert = False
        self._select_lastrowid = not self.compiled.inline and insert_has_identity and (not self.compiled.effective_returning) and (not self._enable_identity_insert) and (not self.executemany)
        if self._enable_identity_insert:
            self.root_connection._cursor_execute(self.cursor, self._opt_encode('SET IDENTITY_INSERT %s ON' % self.identifier_preparer.format_table(tbl)), (), self)