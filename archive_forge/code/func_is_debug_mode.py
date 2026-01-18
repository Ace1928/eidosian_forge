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
@property
def is_debug_mode(self) -> bool:
    """
        Returns the custom debug mode flag from the settings
        """
    if 'is_debug_mode' in self.ctx:
        return self.ctx['is_debug_mode']
    if self.settings and hasattr(self.settings, 'postgres_debug_mode'):
        self.ctx['is_debug_mode'] = getattr(self.settings, 'postgres_debug_mode', False)
        return self.ctx['is_debug_mode']
    if self.config.get('is_debug_mode') is not None:
        self.ctx['is_debug_mode'] = self.config.get('is_debug_mode')
        return self.ctx['is_debug_mode']
    return False