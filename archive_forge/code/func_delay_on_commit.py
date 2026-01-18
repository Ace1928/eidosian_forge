import functools
from django.db import transaction
from celery.app.task import Task
def delay_on_commit(self, *args, **kwargs):
    """Call :meth:`~celery.app.task.Task.delay` with Django's ``on_commit()``."""
    return transaction.on_commit(functools.partial(self.delay, *args, **kwargs))