import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture()
def celery_app(request, celery_config, celery_parameters, celery_enable_logging, use_celery_app_trap):
    """Fixture creating a Celery application instance."""
    mark = request.node.get_closest_marker('celery')
    config = dict(celery_config, **mark.kwargs if mark else {})
    with _create_app(enable_logging=celery_enable_logging, use_trap=use_celery_app_trap, parameters=celery_parameters, **config) as app:
        yield app