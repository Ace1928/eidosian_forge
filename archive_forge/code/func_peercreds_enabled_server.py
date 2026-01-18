import os
import queue
import socket
import tempfile
import threading
import types
import uuid
import urllib.parse  # noqa: WPS301
import pytest
import requests
import requests_unixsocket
from pypytools.gc.custom import DefaultGc
from .._compat import bton, ntob
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS, SYS_PLATFORM
from ..server import IS_UID_GID_RESOLVABLE, Gateway, HTTPServer
from ..workers.threadpool import ThreadPool
from ..testing import (
@pytest.fixture
def peercreds_enabled_server(http_server, unix_sock_file):
    """Construct a test server with ``peercreds_enabled``."""
    httpserver = http_server.send(unix_sock_file)
    httpserver.gateway = _TestGateway
    httpserver.peercreds_enabled = True
    return httpserver