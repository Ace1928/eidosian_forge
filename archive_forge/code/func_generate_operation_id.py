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
def generate_operation_id(*, route: routing.APIRoute, method: str) -> str:
    warnings.warn('fastapi.openapi.utils.generate_operation_id() was deprecated, it is not used internally, and will be removed soon', DeprecationWarning, stacklevel=2)
    if route.operation_id:
        return route.operation_id
    path: str = route.path_format
    return generate_operation_id_for_path(name=route.name, path=path, method=method)