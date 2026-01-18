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
def get_table_info_no_throw(self, table_names: Optional[List[str]]=None) -> str:
    """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
    try:
        return self.get_table_info(table_names)
    except ValueError as e:
        'Format the error message'
        return f'Error: {e}'