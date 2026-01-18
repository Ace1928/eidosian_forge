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
def many_open_sockets(request, resource_limit):
    """Allocate a lot of file descriptors by opening dummy sockets."""
    request.getfixturevalue('_garbage_bin')
    test_sockets = []
    try:
        for _ in range(resource_limit):
            sock = socket.socket()
            test_sockets.append(sock)
            if sock.fileno() >= resource_limit:
                break
        the_highest_fileno = test_sockets[-1].fileno()
        assert the_highest_fileno >= resource_limit
        yield the_highest_fileno
    finally:
        for test_socket in test_sockets:
            test_socket.close()