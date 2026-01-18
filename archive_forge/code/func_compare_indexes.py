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
def compare_indexes(self, metadata_index: Index, reflected_index: Index) -> ComparisonResult:
    msg = []
    unique_msg = self._compare_index_unique(metadata_index, reflected_index)
    if unique_msg:
        msg.append(unique_msg)
    m_exprs = metadata_index.expressions
    r_exprs = reflected_index.expressions
    if len(m_exprs) != len(r_exprs):
        msg.append(f'expression number {len(r_exprs)} to {len(m_exprs)}')
    if msg:
        return ComparisonResult.Different(msg)
    skip = []
    for pos, (m_e, r_e) in enumerate(zip(m_exprs, r_exprs), 1):
        m_compile = self._compile_element(m_e)
        m_text = self._cleanup_index_expr(metadata_index, m_compile)
        r_compile = self._compile_element(r_e)
        r_text = self._cleanup_index_expr(metadata_index, r_compile)
        if m_text == r_text:
            continue
        elif m_compile.strip().endswith('_ops') and (' ' in m_compile or ')' in m_compile):
            skip.append(f'expression #{pos} {m_compile!r} detected as including operator clause.')
            util.warn(f'Expression #{pos} {m_compile!r} in index {reflected_index.name!r} detected to include an operator clause. Expression compare cannot proceed. Please move the operator clause to the ``postgresql_ops`` dict to enable proper compare of the index expressions: https://docs.sqlalchemy.org/en/latest/dialects/postgresql.html#operator-classes')
        else:
            msg.append(f'expression #{pos} {r_compile!r} to {m_compile!r}')
    m_options = self._dialect_options(metadata_index)
    r_options = self._dialect_options(reflected_index)
    if m_options != r_options:
        msg.extend(f'options {r_options} to {m_options}')
    if msg:
        return ComparisonResult.Different(msg)
    elif skip:
        return ComparisonResult.Skip(skip)
    else:
        return ComparisonResult.Equal()