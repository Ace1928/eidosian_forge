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
@property
def session_ro(self) -> async_sessionmaker[AsyncSession]:
    """
        Returns the readonly session
        """
    if self._session_ro is None:
        assert self.config, 'The config must be set'
        self._session_ro = async_sessionmaker(self.engine_ro, class_=AsyncSession, **self.config.get_session_kwargs(readonly=True, **self._kwargs))
    return self._session_ro