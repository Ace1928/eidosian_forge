from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
def signal_notify(source, condition):
    if condition & GLib.IO_IN:
        try:
            return bool(read_socket.recv(1))
        except EnvironmentError as e:
            print(e)
            return False
        return True
    else:
        return False