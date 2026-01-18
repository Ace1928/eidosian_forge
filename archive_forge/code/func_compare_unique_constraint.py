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
def compare_unique_constraint(self, metadata_constraint: UniqueConstraint, reflected_constraint: UniqueConstraint) -> ComparisonResult:
    metadata_tup = self._create_metadata_constraint_sig(metadata_constraint)
    reflected_tup = self._create_reflected_constraint_sig(reflected_constraint)
    meta_sig = metadata_tup.unnamed
    conn_sig = reflected_tup.unnamed
    if conn_sig != meta_sig:
        return ComparisonResult.Different(f'expression {conn_sig} to {meta_sig}')
    metadata_do = self._dialect_options(metadata_tup.const)
    conn_do = self._dialect_options(reflected_tup.const)
    if metadata_do != conn_do:
        return ComparisonResult.Different(f'expression {conn_do} to {metadata_do}')
    return ComparisonResult.Equal()