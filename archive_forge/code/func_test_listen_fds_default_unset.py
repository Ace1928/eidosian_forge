import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds_default_unset():
    os.environ['LISTEN_FDS'] = '1'
    os.environ['LISTEN_PID'] = str(os.getpid())
    assert listen_fds(False) == [3]
    assert listen_fds() == [3]
    assert listen_fds() == []