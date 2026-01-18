import inspect
from contextlib import AsyncExitStack, contextmanager
from copy import deepcopy
from typing import (
import anyio
from fastapi import params
from fastapi._compat import (
from fastapi.background import BackgroundTasks
from fastapi.concurrency import (
from fastapi.dependencies.models import Dependant, SecurityRequirement
from fastapi.logger import logger
from fastapi.security.base import SecurityBase
from fastapi.security.oauth2 import OAuth2, SecurityScopes
from fastapi.security.open_id_connect_url import OpenIdConnect
from fastapi.utils import create_response_field, get_path_param_names
from pydantic.fields import FieldInfo
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, Headers, QueryParams, UploadFile
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response
from starlette.websockets import WebSocket
from typing_extensions import Annotated, get_args, get_origin
def request_params_to_args(required_params: Sequence[ModelField], received_params: Union[Mapping[str, Any], QueryParams, Headers]) -> Tuple[Dict[str, Any], List[Any]]:
    values = {}
    errors = []
    for field in required_params:
        if is_scalar_sequence_field(field) and isinstance(received_params, (QueryParams, Headers)):
            value = received_params.getlist(field.alias) or field.default
        else:
            value = received_params.get(field.alias)
        field_info = field.field_info
        assert isinstance(field_info, params.Param), 'Params must be subclasses of Param'
        loc = (field_info.in_.value, field.alias)
        if value is None:
            if field.required:
                errors.append(get_missing_field_error(loc=loc))
            else:
                values[field.name] = deepcopy(field.default)
            continue
        v_, errors_ = field.validate(value, values, loc=loc)
        if isinstance(errors_, ErrorWrapper):
            errors.append(errors_)
        elif isinstance(errors_, list):
            new_errors = _regenerate_error_with_loc(errors=errors_, loc_prefix=())
            errors.extend(new_errors)
        else:
            values[field.name] = v_
    return (values, errors)