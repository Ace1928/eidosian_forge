from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
def reflect_table(self, table: sa_schema.Table, include_columns: Optional[Collection[str]], exclude_columns: Collection[str]=(), resolve_fks: bool=True, _extend_on: Optional[Set[sa_schema.Table]]=None, _reflect_info: Optional[_ReflectionInfo]=None) -> None:
    """Given a :class:`_schema.Table` object, load its internal
        constructs based on introspection.

        This is the underlying method used by most dialects to produce
        table reflection.  Direct usage is like::

            from sqlalchemy import create_engine, MetaData, Table
            from sqlalchemy import inspect

            engine = create_engine('...')
            meta = MetaData()
            user_table = Table('user', meta)
            insp = inspect(engine)
            insp.reflect_table(user_table, None)

        .. versionchanged:: 1.4 Renamed from ``reflecttable`` to
           ``reflect_table``

        :param table: a :class:`~sqlalchemy.schema.Table` instance.
        :param include_columns: a list of string column names to include
          in the reflection process.  If ``None``, all columns are reflected.

        """
    if _extend_on is not None:
        if table in _extend_on:
            return
        else:
            _extend_on.add(table)
    dialect = self.bind.dialect
    with self._operation_context() as conn:
        schema = conn.schema_for_object(table)
    table_name = table.name
    reflection_options = {k: table.dialect_kwargs.get(k) for k in dialect.reflection_options if k in table.dialect_kwargs}
    table_key = (schema, table_name)
    if _reflect_info is None or table_key not in _reflect_info.columns:
        _reflect_info = self._get_reflection_info(schema, filter_names=[table_name], kind=ObjectKind.ANY, scope=ObjectScope.ANY, _reflect_info=_reflect_info, **table.dialect_kwargs)
    if table_key in _reflect_info.unreflectable:
        raise _reflect_info.unreflectable[table_key]
    if table_key not in _reflect_info.columns:
        raise exc.NoSuchTableError(table_name)
    if _reflect_info.table_options:
        tbl_opts = _reflect_info.table_options.get(table_key)
        if tbl_opts:
            table._validate_dialect_kwargs(tbl_opts)
    found_table = False
    cols_by_orig_name: Dict[str, sa_schema.Column[Any]] = {}
    for col_d in _reflect_info.columns[table_key]:
        found_table = True
        self._reflect_column(table, col_d, include_columns, exclude_columns, cols_by_orig_name)
    if not found_table and (not self.has_table(table_name, schema)):
        raise exc.NoSuchTableError(table_name)
    self._reflect_pk(_reflect_info, table_key, table, cols_by_orig_name, exclude_columns)
    self._reflect_fk(_reflect_info, table_key, table, cols_by_orig_name, include_columns, exclude_columns, resolve_fks, _extend_on, reflection_options)
    self._reflect_indexes(_reflect_info, table_key, table, cols_by_orig_name, include_columns, exclude_columns, reflection_options)
    self._reflect_unique_constraints(_reflect_info, table_key, table, cols_by_orig_name, include_columns, exclude_columns, reflection_options)
    self._reflect_check_constraints(_reflect_info, table_key, table, cols_by_orig_name, include_columns, exclude_columns, reflection_options)
    self._reflect_table_comment(_reflect_info, table_key, table, reflection_options)