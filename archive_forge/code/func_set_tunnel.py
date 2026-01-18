import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def set_tunnel(self, host, port=None, headers=None):
    """Set up host and port for HTTP CONNECT tunnelling.

        In a connection that uses HTTP CONNECT tunneling, the host passed to the
        constructor is used as a proxy server that relays all communication to
        the endpoint passed to `set_tunnel`. This done by sending an HTTP
        CONNECT request to the proxy server when the connection is established.

        This method must be called before the HTTP connection has been
        established.

        The headers argument should be a mapping of extra HTTP headers to send
        with the CONNECT request.
        """
    if self.sock:
        raise RuntimeError("Can't set up tunnel for established connection")
    self._tunnel_host, self._tunnel_port = self._get_hostport(host, port)
    if headers:
        self._tunnel_headers = headers
    else:
        self._tunnel_headers.clear()