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
def instance_ip_grouping_key() -> Dict[str, Any]:
    """Grouping key with instance set to the IP Address of this host."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
        if sys.platform == 'darwin':
            s.connect(('10.255.255.255', 1))
        else:
            s.connect(('localhost', 0))
        return {'instance': s.getsockname()[0]}