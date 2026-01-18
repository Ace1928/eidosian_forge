import collections
import contextlib
import io
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import HTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from . import base_events
from . import events
from . import futures
from . import selectors
from . import tasks
from .coroutines import coroutine
from .log import logger
@contextlib.contextmanager
def run_test_unix_server(*, use_ssl=False):
    with unix_socket_path() as path:
        yield from _run_test_server(address=path, use_ssl=use_ssl, server_cls=SilentUnixWSGIServer, server_ssl_cls=UnixSSLWSGIServer)