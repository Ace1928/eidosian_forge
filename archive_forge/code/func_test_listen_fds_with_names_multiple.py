import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_listen_fds_with_names_multiple():
    os.environ['LISTEN_FDS'] = '3'
    os.environ['LISTEN_PID'] = str(os.getpid())
    os.environ['LISTEN_FDNAMES'] = 'cmds:data:errs'
    assert listen_fds_with_names(False) == {3: 'cmds', 4: 'data', 5: 'errs'}
    assert listen_fds_with_names(True) == {3: 'cmds', 4: 'data', 5: 'errs'}
    assert listen_fds_with_names() == {}