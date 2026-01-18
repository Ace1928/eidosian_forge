from collections import deque
from copy import copy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import (
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel, create_model
from pydantic.version import VERSION as PYDANTIC_VERSION
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin
def get_missing_field_error(loc: Tuple[str, ...]) -> Dict[str, Any]:
    missing_field_error = ErrorWrapper(MissingError(), loc=loc)
    new_error = ValidationError([missing_field_error], RequestErrorModel)
    return new_error.errors()[0]