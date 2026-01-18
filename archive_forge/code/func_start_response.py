import io
import math
import sys
import typing
import warnings
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette.types import Receive, Scope, Send
def start_response(self, status: str, response_headers: typing.List[typing.Tuple[str, str]], exc_info: typing.Any=None) -> None:
    self.exc_info = exc_info
    if not self.response_started:
        self.response_started = True
        status_code_string, _ = status.split(' ', 1)
        status_code = int(status_code_string)
        headers = [(name.strip().encode('ascii').lower(), value.strip().encode('ascii')) for name, value in response_headers]
        anyio.from_thread.run(self.stream_send.send, {'type': 'http.response.start', 'status': status_code, 'headers': headers})