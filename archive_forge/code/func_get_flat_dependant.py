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
def get_flat_dependant(dependant: Dependant, *, skip_repeats: bool=False, visited: Optional[List[CacheKey]]=None) -> Dependant:
    if visited is None:
        visited = []
    visited.append(dependant.cache_key)
    flat_dependant = Dependant(path_params=dependant.path_params.copy(), query_params=dependant.query_params.copy(), header_params=dependant.header_params.copy(), cookie_params=dependant.cookie_params.copy(), body_params=dependant.body_params.copy(), security_schemes=dependant.security_requirements.copy(), use_cache=dependant.use_cache, path=dependant.path)
    for sub_dependant in dependant.dependencies:
        if skip_repeats and sub_dependant.cache_key in visited:
            continue
        flat_sub = get_flat_dependant(sub_dependant, skip_repeats=skip_repeats, visited=visited)
        flat_dependant.path_params.extend(flat_sub.path_params)
        flat_dependant.query_params.extend(flat_sub.query_params)
        flat_dependant.header_params.extend(flat_sub.header_params)
        flat_dependant.cookie_params.extend(flat_sub.cookie_params)
        flat_dependant.body_params.extend(flat_sub.body_params)
        flat_dependant.security_requirements.extend(flat_sub.security_requirements)
    return flat_dependant