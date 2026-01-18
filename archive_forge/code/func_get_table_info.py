from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union
import sqlalchemy
from langchain_core._api import deprecated
from langchain_core.utils import get_from_env
from sqlalchemy import (
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
def get_table_info(self, table_names: Optional[List[str]]=None) -> str:
    """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
    all_table_names = self.get_usable_table_names()
    if table_names is not None:
        missing_tables = set(table_names).difference(all_table_names)
        if missing_tables:
            raise ValueError(f'table_names {missing_tables} not found in database')
        all_table_names = table_names
    metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
    to_reflect = set(all_table_names) - set(metadata_table_names)
    if to_reflect:
        self._metadata.reflect(views=self._view_support, bind=self._engine, only=list(to_reflect), schema=self._schema)
    meta_tables = [tbl for tbl in self._metadata.sorted_tables if tbl.name in set(all_table_names) and (not (self.dialect == 'sqlite' and tbl.name.startswith('sqlite_')))]
    tables = []
    for table in meta_tables:
        if self._custom_table_info and table.name in self._custom_table_info:
            tables.append(self._custom_table_info[table.name])
            continue
        for k, v in table.columns.items():
            if type(v.type) is NullType:
                table._columns.remove(v)
        create_table = str(CreateTable(table).compile(self._engine))
        table_info = f'{create_table.rstrip()}'
        has_extra_info = self._indexes_in_table_info or self._sample_rows_in_table_info
        if has_extra_info:
            table_info += '\n\n/*'
        if self._indexes_in_table_info:
            table_info += f'\n{self._get_table_indexes(table)}\n'
        if self._sample_rows_in_table_info:
            table_info += f'\n{self._get_sample_rows(table)}\n'
        if has_extra_info:
            table_info += '*/'
        tables.append(table_info)
    tables.sort()
    final_str = '\n\n'.join(tables)
    return final_str