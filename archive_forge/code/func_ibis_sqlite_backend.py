import contextlib
import pytest
from panel.tests.conftest import (  # noqa
@pytest.fixture
def ibis_sqlite_backend():
    try:
        import ibis
    except ImportError:
        yield None
    else:
        ibis.set_backend('sqlite')
        yield
        ibis.set_backend(None)