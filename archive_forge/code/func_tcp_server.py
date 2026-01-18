import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
def tcp_server(self, server_prog, *, family=socket.AF_INET, addr=None, timeout=5, backlog=1, max_clients=10):
    if addr is None:
        if family == socket.AF_UNIX:
            with tempfile.NamedTemporaryFile() as tmp:
                addr = tmp.name
        else:
            addr = ('127.0.0.1', 0)
    sock = socket.socket(family, socket.SOCK_STREAM)
    if timeout is None:
        raise RuntimeError('timeout is required')
    if timeout <= 0:
        raise RuntimeError('only blocking sockets are supported')
    sock.settimeout(timeout)
    try:
        sock.bind(addr)
        sock.listen(backlog)
    except OSError as ex:
        sock.close()
        raise ex
    return TestThreadedServer(self, sock, server_prog, timeout, max_clients)