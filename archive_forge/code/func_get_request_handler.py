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
def get_request_handler(dependant: Dependant, body_field: Optional[ModelField]=None, status_code: Optional[int]=None, response_class: Union[Type[Response], DefaultPlaceholder]=Default(JSONResponse), response_field: Optional[ModelField]=None, response_model_include: Optional[IncEx]=None, response_model_exclude: Optional[IncEx]=None, response_model_by_alias: bool=True, response_model_exclude_unset: bool=False, response_model_exclude_defaults: bool=False, response_model_exclude_none: bool=False, dependency_overrides_provider: Optional[Any]=None) -> Callable[[Request], Coroutine[Any, Any, Response]]:
    assert dependant.call is not None, 'dependant.call must be a function'
    is_coroutine = asyncio.iscoroutinefunction(dependant.call)
    is_body_form = body_field and isinstance(body_field.field_info, params.Form)
    if isinstance(response_class, DefaultPlaceholder):
        actual_response_class: Type[Response] = response_class.value
    else:
        actual_response_class = response_class

    async def app(request: Request) -> Response:
        response: Union[Response, None] = None
        async with AsyncExitStack() as file_stack:
            try:
                body: Any = None
                if body_field:
                    if is_body_form:
                        body = await request.form()
                        file_stack.push_async_callback(body.close)
                    else:
                        body_bytes = await request.body()
                        if body_bytes:
                            json_body: Any = Undefined
                            content_type_value = request.headers.get('content-type')
                            if not content_type_value:
                                json_body = await request.json()
                            else:
                                message = email.message.Message()
                                message['content-type'] = content_type_value
                                if message.get_content_maintype() == 'application':
                                    subtype = message.get_content_subtype()
                                    if subtype == 'json' or subtype.endswith('+json'):
                                        json_body = await request.json()
                            if json_body != Undefined:
                                body = json_body
                            else:
                                body = body_bytes
            except json.JSONDecodeError as e:
                validation_error = RequestValidationError([{'type': 'json_invalid', 'loc': ('body', e.pos), 'msg': 'JSON decode error', 'input': {}, 'ctx': {'error': e.msg}}], body=e.doc)
                raise validation_error from e
            except HTTPException:
                raise
            except Exception as e:
                http_error = HTTPException(status_code=400, detail='There was an error parsing the body')
                raise http_error from e
            errors: List[Any] = []
            async with AsyncExitStack() as async_exit_stack:
                solved_result = await solve_dependencies(request=request, dependant=dependant, body=body, dependency_overrides_provider=dependency_overrides_provider, async_exit_stack=async_exit_stack)
                values, errors, background_tasks, sub_response, _ = solved_result
                if not errors:
                    raw_response = await run_endpoint_function(dependant=dependant, values=values, is_coroutine=is_coroutine)
                    if isinstance(raw_response, Response):
                        if raw_response.background is None:
                            raw_response.background = background_tasks
                        response = raw_response
                    else:
                        response_args: Dict[str, Any] = {'background': background_tasks}
                        current_status_code = status_code if status_code else sub_response.status_code
                        if current_status_code is not None:
                            response_args['status_code'] = current_status_code
                        if sub_response.status_code:
                            response_args['status_code'] = sub_response.status_code
                        content = await serialize_response(field=response_field, response_content=raw_response, include=response_model_include, exclude=response_model_exclude, by_alias=response_model_by_alias, exclude_unset=response_model_exclude_unset, exclude_defaults=response_model_exclude_defaults, exclude_none=response_model_exclude_none, is_coroutine=is_coroutine)
                        response = actual_response_class(content, **response_args)
                        if not is_body_allowed_for_status_code(response.status_code):
                            response.body = b''
                        response.headers.raw.extend(sub_response.headers.raw)
            if errors:
                validation_error = RequestValidationError(_normalize_errors(errors), body=body)
                raise validation_error
        if response is None:
            raise FastAPIError("No response object was returned. There's a high chance that the application code is raising an exception and a dependency with yield has a block with a bare except, or a block with except Exception, and is not raising the exception again. Read more about it in the docs: https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#dependencies-with-yield-and-except")
        return response
    return app