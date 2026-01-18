import base64
from contextlib import closing
import gzip
from http.server import BaseHTTPRequestHandler
import os
import socket
from socketserver import ThreadingMixIn
import ssl
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import (
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from .openmetrics import exposition as openmetrics
from .registry import CollectorRegistry, REGISTRY
from .utils import floatToGoString
from .asgi import make_asgi_app  # noqa
def start_wsgi_server(port: int, addr: str='0.0.0.0', registry: CollectorRegistry=REGISTRY, certfile: Optional[str]=None, keyfile: Optional[str]=None, client_cafile: Optional[str]=None, client_capath: Optional[str]=None, protocol: int=ssl.PROTOCOL_TLS_SERVER, client_auth_required: bool=False) -> Tuple[WSGIServer, threading.Thread]:
    """Starts a WSGI server for prometheus metrics as a daemon thread."""

    class TmpServer(ThreadingWSGIServer):
        """Copy of ThreadingWSGIServer to update address_family locally"""
    TmpServer.address_family, addr = _get_best_family(addr, port)
    app = make_wsgi_app(registry)
    httpd = make_server(addr, port, app, TmpServer, handler_class=_SilentHandler)
    if certfile and keyfile:
        context = _get_ssl_ctx(certfile, keyfile, protocol, client_cafile, client_capath, client_auth_required)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    t = threading.Thread(target=httpd.serve_forever)
    t.daemon = True
    t.start()
    return (httpd, t)