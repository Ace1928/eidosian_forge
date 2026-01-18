import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds_no_fds():
    os.unsetenv('LISTEN_FDS')
    os.unsetenv('LISTEN_PID')
    assert listen_fds() == []
    assert listen_fds(True) == []
    assert listen_fds(False) == []