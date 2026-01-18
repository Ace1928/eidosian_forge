import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
@classmethod
def start_response_handler(cls, response, num=1, block_send=None):
    ready_event = threading.Event()

    def socket_handler(listener):
        for _ in range(num):
            ready_event.set()
            sock = listener.accept()[0]
            consume_socket(sock)
            if block_send:
                block_send.wait()
                block_send.clear()
            sock.send(response)
            sock.close()
    cls._start_server(socket_handler)
    return ready_event