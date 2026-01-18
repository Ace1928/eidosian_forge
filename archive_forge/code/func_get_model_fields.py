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
def get_model_fields(obj: DeclarativeMeta) -> Dict[str, Any]:
    """
    Return a dictionary representation of a sqlalchemy model
    """
    fields = {}
    for c in inspect(obj).mapper.column_attrs:
        fields[c.key] = (cast_to_optional(type(getattr(obj, c.key))), Field(None))
    return fields