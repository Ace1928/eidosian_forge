import threading
import time
import pytest
from .._compat import IS_MACOS, IS_WINDOWS  # noqa: WPS436
from ..server import Gateway, HTTPServer
from ..testing import (  # noqa: F401  # pylint: disable=unused-import
from ..testing import get_server_client
def start_srv():
    bind_addr = (yield)
    if bind_addr is None:
        return
    httpserver = make_http_server(bind_addr)
    yield httpserver
    yield httpserver