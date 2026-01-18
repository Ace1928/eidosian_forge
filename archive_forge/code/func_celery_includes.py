import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture(scope='session')
def celery_includes():
    """You can override this include modules when a worker start.

    You can have this return a list of module names to import,
    these can be task modules, modules registering signals, and so on.
    """
    return ()