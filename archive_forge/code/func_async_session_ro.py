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
def async_session_ro(self) -> Optional[AsyncSession]:
    """
        Returns the read-only async session
        """
    if self.asess_ro is None and self.has_ro:
        sess_args = self.config.get('async_session_ro_args', {})
        if (class_ := get_nested_arg(sess_args, self.config, 'class_', AsyncSession)):
            if isinstance(class_, str):
                class_ = import_string(class_)
            sess_args['class_'] = class_
        if (event_hooks := get_nested_arg(sess_args, self.config, 'event_hooks', [])):
            self.ctx['async_session_ro_hooks'] = event_hooks
        if (is_scoped_session := get_nested_arg(sess_args, self.config, 'is_scoped_session', False)):
            self.ctx['is_scoped_async_session_ro'] = is_scoped_session
        self.asess_ro = sessionmaker(bind=self.async_engine_ro, autoflush=get_nested_arg(sess_args, self.config, 'autoflush', True), expire_on_commit=get_nested_arg(sess_args, self.config, 'expire_on_commit', False), **sess_args)
        if (is_scoped_session := self.ctx.get('is_scoped_async_session_ro')):
            self.asess_ro = async_scoped_session(self.asess_ro, scopefunc=asyncio.current_task)
    return self.asess_ro if self.asess_ro is not None else None