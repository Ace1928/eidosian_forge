from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
@classmethod
def from_get_request(cls, **kwargs) -> 'BaseRequest':
    """
        Builds the BaseRequest from the GET Request
        """
    from lazyops.utils.helpers import build_dict_from_query
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    for key in kwargs:
        if not cls.model_fields.get(key):
            continue
        annotation = cls.model_fields[key].annotation
        if isinstance(kwargs[key], str) and (annotation.__origin__ == dict or (annotation.__origin__ == Union and 'typing.Dict' in str(annotation.__args__[0]))):
            kwargs[key] = build_dict_from_query(kwargs[key])
    return cls(**kwargs)