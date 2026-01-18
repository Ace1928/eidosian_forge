import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture(scope='session')
def celery_session_worker(request, celery_session_app, celery_includes, celery_class_tasks, celery_worker_pool, celery_worker_parameters):
    """Session Fixture: Start worker that lives throughout test suite."""
    from .testing import worker
    if not NO_WORKER:
        for module in celery_includes:
            celery_session_app.loader.import_task_module(module)
        for class_task in celery_class_tasks:
            celery_session_app.register_task(class_task)
        with worker.start_worker(celery_session_app, pool=celery_worker_pool, **celery_worker_parameters) as w:
            yield w