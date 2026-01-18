import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
@contextlib.contextmanager
def closing_socketpair(family):
    pair = socket.socketpair(family)
    try:
        yield pair
    finally:
        pair[0].close()
        pair[1].close()