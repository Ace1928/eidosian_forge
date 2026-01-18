import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test__is_fifo(tmpdir):
    path = tmpdir.join('test.fifo').strpath
    posix.mkfifo(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    assert _is_fifo(fd, None)
    assert _is_fifo(fd, path)