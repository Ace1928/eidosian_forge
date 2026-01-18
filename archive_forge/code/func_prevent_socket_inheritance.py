import io
import os
import socket
import threading
import time
import selectors
from contextlib import suppress
from . import errors
from ._compat import IS_WINDOWS
from .makefile import MakeFile
def prevent_socket_inheritance(sock):
    """Stub inheritance prevention.

            Dummy function, since neither fcntl nor ctypes are available.
            """
    pass