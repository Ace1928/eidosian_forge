import contextlib
import os
import platform
import socket
import sys
import textwrap
import typing  # noqa: F401
import unittest
import warnings
from tornado.testing import bind_unused_port
def refusing_port():
    """Returns a local port number that will refuse all connections.

    Return value is (cleanup_func, port); the cleanup function
    must be called to free the port to be reused.
    """
    server_socket, port = bind_unused_port()
    server_socket.setblocking(True)
    client_socket = socket.socket()
    client_socket.connect(('127.0.0.1', port))
    conn, client_addr = server_socket.accept()
    conn.close()
    server_socket.close()
    return (client_socket.close, client_addr[1])