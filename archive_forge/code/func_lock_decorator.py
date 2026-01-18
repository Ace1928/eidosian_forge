import socket
import sys
import threading
import time
from . import Adapter
from .. import errors, server as cheroot_server
from ..makefile import StreamReader, StreamWriter
def lock_decorator(method):
    """Create a proxy method for a new class."""

    def proxy_wrapper(self, *args):
        self._lock.acquire()
        try:
            new_args = args[:] if method not in proxy_methods_no_args else []
            return getattr(self._ssl_conn, method)(*new_args)
        finally:
            self._lock.release()
    return proxy_wrapper