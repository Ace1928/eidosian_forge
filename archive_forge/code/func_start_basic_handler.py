import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
@classmethod
def start_basic_handler(cls, **kw):
    return cls.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n', **kw)