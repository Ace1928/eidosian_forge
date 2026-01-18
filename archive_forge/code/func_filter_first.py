import contextlib
from sqlalchemy import func, text
from sqlalchemy import delete as sqlalchemy_delete
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy import exists as sqlalchemy_exists
from sqlalchemy.future import select
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload, joinedload, immediateload
from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, Boolean, Identity, Enum
from typing import Any, Generator, AsyncGenerator, Iterable, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from lazyops.utils import create_unique_id, create_timestamp
from lazyops.utils.logs import logger
from lazyops.types import lazyproperty
from lazyops.libs.psqldb.base import Base, PostgresDB, AsyncSession, Session
from lazyops.libs.psqldb.utils import SQLJson, get_pydantic_model, object_serializer, get_sqlmodel_dict
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
@classmethod
def filter_first(cls, attr: str, value: Any, model_type: Optional[ModelType]=None, readonly: Optional[bool]=False) -> Optional[ModelType]:
    model_type = model_type or cls
    attr_obj = getattr(model_type, attr)
    stmt = select(model_type).where(attr_obj == value)
    with PostgresDB.session(ro=readonly) as db_sess:
        return cls.first(db_sess, stmt, model_type)