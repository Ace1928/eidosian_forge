import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test__is_fifo_bad_fd(tmpdir):
    path = tmpdir.join('test.fifo').strpath
    with pytest.raises(OSError):
        assert not _is_fifo(-1, None)
    with pytest.raises(OSError):
        assert not _is_fifo(-1, path)