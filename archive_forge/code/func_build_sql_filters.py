from __future__ import annotations
import gc
import abc
import asyncio
import datetime
import contextlib
from pathlib import Path
from pydantic.networks import PostgresDsn
from pydantic_settings import BaseSettings
from pydantic import validator, model_validator, computed_field, BaseModel, Field, PrivateAttr
from sqlalchemy import text as sql_text, TextClause
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from lazyops.utils.logs import logger, Logger
from lazyops.utils.lazy import lazy_import
from ...utils.helpers import update_dict
from typing import Any, Dict, List, Optional, Type, Literal, Iterable, Tuple, TypeVar, Union, Annotated, Callable, Generator, AsyncGenerator, Set, TYPE_CHECKING
def build_sql_filters(self, and_filters: Optional[Dict[str, Union[int, float, datetime.datetime, Dict, List, Any]]]=None, or_filters: Optional[Dict[str, Union[int, float, datetime.datetime, Dict, List, Any]]]=None) -> List[Dict[str, Union[List[str], str]]]:
    """
        Creates the proper SQL filters
        [
            {
                "conditional": "AND",
                "statements": [
                    "statement1",
                    "statement2",
                    "statement3",
                ]
            },
            {
                "conditional": "OR",
                "statements": [
                    "statement4",
                    "statement5",
                    "statement6",
                ]
            }
        ]
        which is properly formatted for the template
        """
    filters = []
    if and_filters:
        filters.append(self.build_sql_filter(conditional='AND', **and_filters))
    if or_filters:
        filters.append(self.build_sql_filter(conditional='OR', **or_filters))
    return filters