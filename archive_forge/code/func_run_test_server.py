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
def run_test_server(*, host='127.0.0.1', port=0, use_ssl=False):
    yield from _run_test_server(address=(host, port), use_ssl=use_ssl, server_cls=SilentWSGIServer, server_ssl_cls=SSLWSGIServer)