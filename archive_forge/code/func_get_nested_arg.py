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
def get_nested_arg(config: Dict[str, Any], base_config: Dict[str, Any], key: str, default: Any=None) -> Any:
    """
    Helper to get a nested value from a config
    """
    return config.pop(key) if key in config else base_config.get(key, default)