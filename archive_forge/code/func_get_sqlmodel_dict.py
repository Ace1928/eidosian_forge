import json
import typing
import datetime
import contextlib
from enum import Enum
from sqlalchemy import inspect
from lazyops.utils.serialization import object_serializer, Json
from sqlalchemy.ext.declarative import DeclarativeMeta
from pydantic import create_model, BaseModel, Field
from typing import Optional, Dict, Any, List, Union, Type, cast
def get_sqlmodel_dict(obj: DeclarativeMeta) -> Dict[str, Any]:
    """
    Return a dictionary representation of a sqlalchemy model
    """
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}