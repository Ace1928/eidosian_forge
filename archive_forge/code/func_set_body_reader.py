import io
import re
import socket
from gunicorn.http.body import ChunkedReader, LengthReader, EOFReader, Body
from gunicorn.http.errors import (
from gunicorn.http.errors import InvalidProxyLine, ForbiddenProxyRequest
from gunicorn.http.errors import InvalidSchemeHeaders
from gunicorn.util import bytes_to_str, split_request_uri
def set_body_reader(self):
    super().set_body_reader()
    if isinstance(self.body.reader, EOFReader):
        self.body = Body(LengthReader(self.unreader, 0))