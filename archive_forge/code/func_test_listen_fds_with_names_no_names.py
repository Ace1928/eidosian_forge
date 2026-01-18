import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds_with_names_no_names():
    os.environ['LISTEN_FDS'] = '1'
    os.environ['LISTEN_PID'] = str(os.getpid())
    os.unsetenv('LISTEN_FDNAMES')
    assert listen_fds_with_names(False) == {3: 'unknown'}
    assert listen_fds_with_names(True) == {3: 'unknown'}
    assert listen_fds_with_names() == {}