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
@model_validator(mode='after')
def post_init_validate(self):
    """
        Validate after init
        """
    if not self.superuser_url:
        self.superuser_url = self.url
    if isinstance(self.engine_poolclass, str):
        self.engine_poolclass = lazy_import(self.engine_poolclass)
    if isinstance(self.engine_json_serializer, str):
        if self.engine_json_serializer == 'json':
            try:
                from kvdb.io.serializers import get_serializer
                serializer = get_serializer('json')
                self.engine_json_serializer = serializer.dumps
            except ImportError:
                from lazyops.utils.serialization import Json
                self.engine_json_serializer = Json.dumps
        else:
            try:
                self.engine_json_serializer = lazy_import(self.engine_json_serializer)
            except Exception as e:
                logger.error(f'Failed to import the JSON Serializer: {e}')
    if not self.engine_kwargs:
        self.engine_kwargs['pool_pre_ping'] = True
    if not self.session_rw_kwargs:
        self.session_rw_kwargs['expire_on_commit'] = False
    if not self.session_ro_kwargs:
        self.session_ro_kwargs['autoflush'] = False
        self.session_ro_kwargs['autocommit'] = False
    return self