import os
import time
import asyncio
import contextlib
from lazyops.imports._sqlalchemy import require_sql
from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_scoped_session
from lazyops.utils.logs import logger
from lazyops.utils import Json
from lazyops.types import BaseModel, lazyproperty, BaseSettings, Field
from typing import Any, Generator, AsyncGenerator, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from pydantic.networks import PostgresDsn
from lazyops.libs.psqldb.retry import reconnecting_engine
from lazyops.utils.helpers import import_string
def get_admin_uri(cls, host: Optional[str]=None, port: Optional[int]=None, user: Optional[str]=None, password: Optional[str]=None, db: Optional[str]=None) -> SafePostgresDsn:
    """
        Returns the admin uri
        """
    uri = f'{host or cls.uri.host}:{port or cls.uri.port}/{db or cls.pg_admin_db}'
    auth = f'{user or cls.pg_admin_user}'
    if password or cls.pg_admin_password:
        auth += f':{password or cls.pg_admin_password}'
    uri = f'{auth}@{uri}'
    return uri_builder(uri, scheme=cls.scheme)