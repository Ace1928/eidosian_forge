import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_booted():
    if os.path.exists('/run/systemd/system'):
        assert booted()
    else:
        assert booted() in {False, True}