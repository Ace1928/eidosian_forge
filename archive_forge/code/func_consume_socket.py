import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
def consume_socket(sock, chunks=65536):
    consumed = bytearray()
    while True:
        b = sock.recv(chunks)
        consumed += b
        if b.endswith(b'\r\n\r\n'):
            break
    return consumed