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
def correct_for_autogen_constraints(self, conn_unique_constraints, conn_indexes, metadata_unique_constraints, metadata_indexes):
    doubled_constraints = {index for index in conn_indexes if index.info.get('duplicates_constraint')}
    for ix in doubled_constraints:
        conn_indexes.remove(ix)
    if not sqla_compat.sqla_2:
        self._skip_functional_indexes(metadata_indexes, conn_indexes)