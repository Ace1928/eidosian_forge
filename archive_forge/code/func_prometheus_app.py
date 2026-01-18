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
def prometheus_app(environ, start_response):
    accept_header = environ.get('HTTP_ACCEPT')
    accept_encoding_header = environ.get('HTTP_ACCEPT_ENCODING')
    params = parse_qs(environ.get('QUERY_STRING', ''))
    if environ['PATH_INFO'] == '/favicon.ico':
        status = '200 OK'
        headers = [('', '')]
        output = b''
    else:
        status, headers, output = _bake_output(registry, accept_header, accept_encoding_header, params, disable_compression)
    start_response(status, headers)
    return [output]