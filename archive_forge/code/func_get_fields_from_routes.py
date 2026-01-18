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
def get_fields_from_routes(routes: Sequence[BaseRoute]) -> List[ModelField]:
    body_fields_from_routes: List[ModelField] = []
    responses_from_routes: List[ModelField] = []
    request_fields_from_routes: List[ModelField] = []
    callback_flat_models: List[ModelField] = []
    for route in routes:
        if getattr(route, 'include_in_schema', None) and isinstance(route, routing.APIRoute):
            if route.body_field:
                assert isinstance(route.body_field, ModelField), 'A request body must be a Pydantic Field'
                body_fields_from_routes.append(route.body_field)
            if route.response_field:
                responses_from_routes.append(route.response_field)
            if route.response_fields:
                responses_from_routes.extend(route.response_fields.values())
            if route.callbacks:
                callback_flat_models.extend(get_fields_from_routes(route.callbacks))
            params = get_flat_params(route.dependant)
            request_fields_from_routes.extend(params)
    flat_models = callback_flat_models + list(body_fields_from_routes + responses_from_routes + request_fields_from_routes)
    return flat_models