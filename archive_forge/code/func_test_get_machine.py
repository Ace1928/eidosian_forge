import contextlib
import errno
import uuid
import pytest
from systemd import id128
def test_get_machine():
    u1 = id128.get_machine()
    u2 = id128.get_machine()
    assert u1 == u2