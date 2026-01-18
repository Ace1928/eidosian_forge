import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def get_socket(self):
    """Return the socket object used internally."""
    return self.sock