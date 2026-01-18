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
def run_no_throw(self, command: str, fetch: Literal['all', 'one']='all', include_columns: bool=False, *, parameters: Optional[Dict[str, Any]]=None, execution_options: Optional[Dict[str, Any]]=None) -> Union[str, Sequence[Dict[str, Any]], Result[Any]]:
    """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
    try:
        return self.run(command, fetch, parameters=parameters, execution_options=execution_options, include_columns=include_columns)
    except SQLAlchemyError as e:
        'Format the error message'
        return f'Error: {e}'