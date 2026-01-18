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
def resource_limit(request):
    """Set the resource limit two times bigger then requested."""
    resource = pytest.importorskip('resource', reason='The "resource" module is Unix-specific')
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (request.param * 2, hard_limit))
    try:
        yield request.param
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))