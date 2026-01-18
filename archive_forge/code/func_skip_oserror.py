import contextlib
import errno
import uuid
import pytest
from systemd import id128
@contextlib.contextmanager
def skip_oserror(code):
    try:
        yield
    except (OSError, IOError) as e:
        if e.errno == code:
            pytest.skip()
        raise