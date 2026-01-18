import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture(scope='session')
def celery_session_app(request, celery_config, celery_parameters, celery_enable_logging, use_celery_app_trap):
    """Session Fixture: Return app for session fixtures."""
    mark = request.node.get_closest_marker('celery')
    config = dict(celery_config, **mark.kwargs if mark else {})
    with _create_app(enable_logging=celery_enable_logging, use_trap=use_celery_app_trap, parameters=celery_parameters, **config) as app:
        if not use_celery_app_trap:
            app.set_default()
            app.set_current()
        yield app