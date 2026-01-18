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
def get_usable_table_names(self) -> Iterable[str]:
    """Get names of tables available."""
    if self._include_tables:
        return sorted(self._include_tables)
    return sorted(self._all_tables - self._ignore_tables)