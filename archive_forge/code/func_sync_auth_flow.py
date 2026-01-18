from __future__ import annotations
import hashlib
import os
import re
import time
import typing
from base64 import b64encode
from urllib.request import parse_http_list
from ._exceptions import ProtocolError
from ._models import Cookies, Request, Response
from ._utils import to_bytes, to_str, unquote
def sync_auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
    """
        Execute the authentication flow synchronously.

        By default, this defers to `.auth_flow()`. You should override this method
        when the authentication scheme does I/O and/or uses concurrency primitives.
        """
    if self.requires_request_body:
        request.read()
    flow = self.auth_flow(request)
    request = next(flow)
    while True:
        response = (yield request)
        if self.requires_response_body:
            response.read()
        try:
            request = flow.send(response)
        except StopIteration:
            break