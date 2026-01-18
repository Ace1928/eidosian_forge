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
def get_definitions(*, fields: List[ModelField], schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, separate_input_output_schemas: bool=True) -> Tuple[Dict[Tuple[ModelField, Literal['validation', 'serialization']], JsonSchemaValue], Dict[str, Dict[str, Any]]]:
    models = get_flat_models_from_fields(fields, known_models=set())
    return ({}, get_model_definitions(flat_models=models, model_name_map=model_name_map))