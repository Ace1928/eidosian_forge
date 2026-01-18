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
def get_websocket_app(dependant: Dependant, dependency_overrides_provider: Optional[Any]=None) -> Callable[[WebSocket], Coroutine[Any, Any, Any]]:

    async def app(websocket: WebSocket) -> None:
        async with AsyncExitStack() as async_exit_stack:
            websocket.scope['fastapi_astack'] = async_exit_stack
            solved_result = await solve_dependencies(request=websocket, dependant=dependant, dependency_overrides_provider=dependency_overrides_provider, async_exit_stack=async_exit_stack)
            values, errors, _, _2, _3 = solved_result
            if errors:
                raise WebSocketRequestValidationError(_normalize_errors(errors))
            assert dependant.call is not None, 'dependant.call must be a function'
            await dependant.call(**values)
    return app