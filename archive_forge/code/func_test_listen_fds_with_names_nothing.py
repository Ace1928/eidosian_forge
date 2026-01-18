import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds_with_names_nothing():
    os.unsetenv('LISTEN_FDS')
    os.unsetenv('LISTEN_PID')
    os.unsetenv('LISTEN_FDNAMES')
    assert listen_fds_with_names() == {}
    assert listen_fds_with_names(True) == {}
    assert listen_fds_with_names(False) == {}