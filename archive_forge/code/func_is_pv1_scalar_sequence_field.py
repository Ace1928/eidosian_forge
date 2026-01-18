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
def is_pv1_scalar_sequence_field(field: ModelField) -> bool:
    if field.shape in sequence_shapes and (not lenient_issubclass(field.type_, BaseModel)):
        if field.sub_fields is not None:
            for sub_field in field.sub_fields:
                if not is_pv1_scalar_field(sub_field):
                    return False
        return True
    if _annotation_is_sequence(field.type_):
        return True
    return False