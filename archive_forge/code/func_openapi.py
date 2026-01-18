from enum import Enum
from typing import (
from fastapi import routing
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.exception_handlers import (
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.logger import logger
from fastapi.openapi.docs import (
from fastapi.openapi.utils import get_openapi
from fastapi.params import Depends
from fastapi.types import DecoratedCallable, IncEx
from fastapi.utils import generate_unique_id
from starlette.applications import Starlette
from starlette.datastructures import State
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from typing_extensions import Annotated, Doc, deprecated  # type: ignore [attr-defined]
def openapi(self) -> Dict[str, Any]:
    """
        Generate the OpenAPI schema of the application. This is called by FastAPI
        internally.

        The first time it is called it stores the result in the attribute
        `app.openapi_schema`, and next times it is called, it just returns that same
        result. To avoid the cost of generating the schema every time.

        If you need to modify the generated OpenAPI schema, you could modify it.

        Read more in the
        [FastAPI docs for OpenAPI](https://fastapi.tiangolo.com/how-to/extending-openapi/).
        """
    if not self.openapi_schema:
        self.openapi_schema = get_openapi(title=self.title, version=self.version, openapi_version=self.openapi_version, summary=self.summary, description=self.description, terms_of_service=self.terms_of_service, contact=self.contact, license_info=self.license_info, routes=self.routes, webhooks=self.webhooks.routes, tags=self.openapi_tags, servers=self.servers, separate_input_output_schemas=self.separate_input_output_schemas)
    return self.openapi_schema