import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture(scope='session')
def celery_config():
    """Redefine this fixture to configure the test Celery app.

    The config returned by your fixture will then be used
    to configure the :func:`celery_app` fixture.
    """
    return {}