import functools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import http.client
import OpenSSL.SSL
import pytest
import requests
import trustme
from .._compat import bton, ntob, ntou
from .._compat import IS_ABOVE_OPENSSL10, IS_CI, IS_PYPY
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS
from ..server import HTTPServer, get_ssl_adapter_class
from ..testing import (
from ..wsgi import Gateway_10
def make_tls_http_server(bind_addr, ssl_adapter, request):
    """Create and start an HTTP server bound to ``bind_addr``."""
    httpserver = HTTPServer(bind_addr=bind_addr, gateway=HelloWorldGateway)
    httpserver.ssl_adapter = ssl_adapter
    threading.Thread(target=httpserver.safe_start).start()
    while not httpserver.ready:
        time.sleep(0.1)
    request.addfinalizer(httpserver.stop)
    return httpserver