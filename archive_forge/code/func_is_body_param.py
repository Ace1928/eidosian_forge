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
def is_body_param(*, param_field: ModelField, is_path_param: bool) -> bool:
    if is_path_param:
        assert is_scalar_field(field=param_field), 'Path params must be of one of the supported types'
        return False
    elif is_scalar_field(field=param_field):
        return False
    elif isinstance(param_field.field_info, (params.Query, params.Header)) and is_scalar_sequence_field(param_field):
        return False
    else:
        assert isinstance(param_field.field_info, params.Body), f'Param: {param_field.name} can only be a request body, using Body()'
        return True