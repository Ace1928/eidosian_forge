import contextlib
import errno
import uuid
import pytest
from systemd import id128
def test_get_boot():
    u1 = id128.get_boot()
    u2 = id128.get_boot()
    assert u1 == u2