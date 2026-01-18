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
def passthrough_redirect_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes) -> Callable[[], None]:
    """
    Handler that automatically trusts redirect responses for all HTTP methods.

    Augments standard HTTPRedirectHandler capability by permitting PUT requests,
    preserving the method upon redirect, and passing through all headers and
    data from the original request. Only use this handler if you control or
    trust the source of redirect responses you encounter when making requests
    via the Prometheus client. This handler will simply repeat the identical
    request, including same method and data, to the new redirect URL."""
    return _make_handler(url, method, timeout, headers, data, _PrometheusRedirectHandler)