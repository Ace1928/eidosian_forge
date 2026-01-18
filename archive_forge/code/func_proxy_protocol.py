import io
import re
import socket
from gunicorn.http.body import ChunkedReader, LengthReader, EOFReader, Body
from gunicorn.http.errors import (
from gunicorn.http.errors import InvalidProxyLine, ForbiddenProxyRequest
from gunicorn.http.errors import InvalidSchemeHeaders
from gunicorn.util import bytes_to_str, split_request_uri
def proxy_protocol(self, line):
    """        Detect, check and parse proxy protocol.

        :raises: ForbiddenProxyRequest, InvalidProxyLine.
        :return: True for proxy protocol line else False
        """
    if not self.cfg.proxy_protocol:
        return False
    if self.req_number != 1:
        return False
    if not line.startswith('PROXY'):
        return False
    self.proxy_protocol_access_check()
    self.parse_proxy_protocol(line)
    return True