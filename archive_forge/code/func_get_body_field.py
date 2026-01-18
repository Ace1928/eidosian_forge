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
def get_body_field(*, dependant: Dependant, name: str) -> Optional[ModelField]:
    flat_dependant = get_flat_dependant(dependant)
    if not flat_dependant.body_params:
        return None
    first_param = flat_dependant.body_params[0]
    field_info = first_param.field_info
    embed = getattr(field_info, 'embed', None)
    body_param_names_set = {param.name for param in flat_dependant.body_params}
    if len(body_param_names_set) == 1 and (not embed):
        check_file_field(first_param)
        return first_param
    for param in flat_dependant.body_params:
        setattr(param.field_info, 'embed', True)
    model_name = 'Body_' + name
    BodyModel = create_body_model(fields=flat_dependant.body_params, model_name=model_name)
    required = any((True for f in flat_dependant.body_params if f.required))
    BodyFieldInfo_kwargs: Dict[str, Any] = {'annotation': BodyModel, 'alias': 'body'}
    if not required:
        BodyFieldInfo_kwargs['default'] = None
    if any((isinstance(f.field_info, params.File) for f in flat_dependant.body_params)):
        BodyFieldInfo: Type[params.Body] = params.File
    elif any((isinstance(f.field_info, params.Form) for f in flat_dependant.body_params)):
        BodyFieldInfo = params.Form
    else:
        BodyFieldInfo = params.Body
        body_param_media_types = [f.field_info.media_type for f in flat_dependant.body_params if isinstance(f.field_info, params.Body)]
        if len(set(body_param_media_types)) == 1:
            BodyFieldInfo_kwargs['media_type'] = body_param_media_types[0]
    final_field = create_response_field(name='body', type_=BodyModel, required=required, alias='body', field_info=BodyFieldInfo(**BodyFieldInfo_kwargs))
    check_file_field(final_field)
    return final_field