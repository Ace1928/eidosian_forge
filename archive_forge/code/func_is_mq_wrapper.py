import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def is_mq_wrapper(arg):
    try:
        return is_mq(arg)
    except OSError as error:
        assert error.errno == errno.EBADF
        return False