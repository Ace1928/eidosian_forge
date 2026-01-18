import http.client
import inspect
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast
from fastapi import routing
from fastapi._compat import (
from fastapi.datastructures import DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant, get_flat_params
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.constants import METHODS_WITH_BODY, REF_PREFIX, REF_TEMPLATE
from fastapi.openapi.models import OpenAPI
from fastapi.params import Body, Param
from fastapi.responses import Response
from fastapi.types import ModelNameMap
from fastapi.utils import (
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from typing_extensions import Literal
def get_openapi_operation_request_body(*, body_field: Optional[ModelField], schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[Tuple[ModelField, Literal['validation', 'serialization']], JsonSchemaValue], separate_input_output_schemas: bool=True) -> Optional[Dict[str, Any]]:
    if not body_field:
        return None
    assert isinstance(body_field, ModelField)
    body_schema = get_schema_from_model_field(field=body_field, schema_generator=schema_generator, model_name_map=model_name_map, field_mapping=field_mapping, separate_input_output_schemas=separate_input_output_schemas)
    field_info = cast(Body, body_field.field_info)
    request_media_type = field_info.media_type
    required = body_field.required
    request_body_oai: Dict[str, Any] = {}
    if required:
        request_body_oai['required'] = required
    request_media_content: Dict[str, Any] = {'schema': body_schema}
    if field_info.openapi_examples:
        request_media_content['examples'] = jsonable_encoder(field_info.openapi_examples)
    elif field_info.example != Undefined:
        request_media_content['example'] = jsonable_encoder(field_info.example)
    request_body_oai['content'] = {request_media_type: request_media_content}
    return request_body_oai