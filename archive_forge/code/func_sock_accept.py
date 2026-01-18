import socket
import sys
import warnings
from . import base_events
from . import constants
from . import futures
from . import sslproto
from . import transports
from .log import logger
def sock_accept(self, sock):
    return self._proactor.accept(sock)