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
def get_model_definitions(*, flat_models: Set[Union[Type[BaseModel], Type[Enum]]], model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str]) -> Dict[str, Any]:
    definitions: Dict[str, Dict[str, Any]] = {}
    for model in flat_models:
        m_schema, m_definitions, m_nested_models = model_process_schema(model, model_name_map=model_name_map, ref_prefix=REF_PREFIX)
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        if 'description' in m_schema:
            m_schema['description'] = m_schema['description'].split('\x0c')[0]
        definitions[model_name] = m_schema
    return definitions