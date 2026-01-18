import socket
import sys
import threading
import time
from . import Adapter
from .. import errors, server as cheroot_server
from ..makefile import StreamReader, StreamWriter
def proxy_prop_wrapper(self):
    return getattr(self._ssl_conn, property_)