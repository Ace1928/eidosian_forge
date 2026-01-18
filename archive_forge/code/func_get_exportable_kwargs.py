from __future__ import annotations
import datetime
from pydantic import BaseModel
from pydantic.alias_generators import to_snake
from dataclasses import dataclass
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import MappedAsDataclass
from sqlalchemy.orm import Mapped
from sqlalchemy import MetaData
from sqlalchemy.orm import mapped_column
from sqlalchemy import Text, Table
from sqlalchemy import func as sql_func
from sqlalchemy.orm import defer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import text, select, Select, ColumnElement, and_, update, Update, delete, or_
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import registry
from ...types import errors
from typing import Optional, Type, TypeVar, Union, Set, Any, Tuple, List, Dict, cast, Generic, Generator, Callable, TYPE_CHECKING
def get_exportable_kwargs(self, include: Any=None, exclude: Any=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, **kwargs) -> Dict[str, Any]:
    """
        Returns the exportable kwargs
        """
    data = {k: v for k, v in self.__dict__.items() if k in self.get_non_relationship_fields(include=include, exclude=exclude, **kwargs)}
    if exclude_none:
        data = {k: v for k, v in data.items() if v is not None}
    if exclude_unset or exclude_defaults:
        data = {k: v for k, v in data.items() if v != self.__mapper__.columns[k].default.arg}
    return data