import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
def part(self, *args, **kwargs):
    result = target(self, *args, **kwargs)
    self.sock.sendall(cls._get_socket_mark(self.sock, False))
    return result