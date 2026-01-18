import asyncio
import dataclasses
import email.message
import inspect
import json
from contextlib import AsyncExitStack
from enum import Enum, IntEnum
from typing import (
from fastapi import params
from fastapi._compat import (
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import (
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import (
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import (
from pydantic import BaseModel
from starlette import routing
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import (
from starlette.routing import Mount as Mount  # noqa
from starlette.types import ASGIApp, Lifespan, Scope
from starlette.websockets import WebSocket
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
    return get_request_handler(dependant=self.dependant, body_field=self.body_field, status_code=self.status_code, response_class=self.response_class, response_field=self.secure_cloned_response_field, response_model_include=self.response_model_include, response_model_exclude=self.response_model_exclude, response_model_by_alias=self.response_model_by_alias, response_model_exclude_unset=self.response_model_exclude_unset, response_model_exclude_defaults=self.response_model_exclude_defaults, response_model_exclude_none=self.response_model_exclude_none, dependency_overrides_provider=self.dependency_overrides_provider)