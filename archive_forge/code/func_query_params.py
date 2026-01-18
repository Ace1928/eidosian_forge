from __future__ import annotations
import json
import typing
from http import cookies as http_cookies
import anyio
from starlette._utils import AwaitableOrContextManager, AwaitableOrContextManagerWrapper
from starlette.datastructures import URL, Address, FormData, Headers, QueryParams, State
from starlette.exceptions import HTTPException
from starlette.formparsers import FormParser, MultiPartException, MultiPartParser
from starlette.types import Message, Receive, Scope, Send
@property
def query_params(self) -> QueryParams:
    if not hasattr(self, '_query_params'):
        self._query_params = QueryParams(self.scope['query_string'])
    return self._query_params