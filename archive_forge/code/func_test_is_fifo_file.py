import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_is_fifo_file(tmpdir):
    file = tmpdir.join('test.fifo')
    file.write('boo')
    path = file.strpath
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    file = os.fdopen(fd, 'r')
    assert not is_fifo(file, None)
    assert not is_fifo(file, path)
    assert not is_fifo(fd, None)
    assert not is_fifo(fd, path)