import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture()
def depends_on_current_app(celery_app):
    """Fixture that sets app as current."""
    celery_app.set_current()