import socket
import sys
import warnings
from . import base_events
from . import constants
from . import futures
from . import sslproto
from . import transports
from .log import logger
def sock_sendall(self, sock, data):
    return self._proactor.send(sock, data)