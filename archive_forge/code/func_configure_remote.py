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
def configure_remote(self, url: str, name: str, readonly_url: Optional[str]=None, superuser_url: Optional[str]=None, overwrite: Optional[bool]=None, **kwargs):
    """
        Configures the remote database
        """
    if name in self.remote_connections and (not overwrite):
        raise ValueError(f'Remote Connection {name} already exists')
    _config = {'name': name, 'url': url, 'readonly_url': readonly_url, 'superuser_url': superuser_url}
    _config = {k: v for k, v in _config.items() if v is not None}
    _config['extra_kws'] = kwargs
    self.remote_connections[name] = RemoteDatabaseConnection.model_validate(_config)