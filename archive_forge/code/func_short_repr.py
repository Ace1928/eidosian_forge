from __future__ import annotations
import enum
import typing
import datetime
import croniter
from aiokeydb.v2.types.base import BaseModel, lazyproperty, Field, validator
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.configs import settings
from aiokeydb.v2.utils.logs import logger
from aiokeydb.v2.types.static import JobStatus, TaskType, TERMINAL_STATUSES, UNSUCCESSFUL_TERMINAL_STATUSES, INCOMPLETE_STATUSES
@property
def short_repr(self):
    """
        Shortened representation of the job.
        """
    kwargs = ', '.join((f'{k}={v}' for k, v in {'id': self.id, 'function': self.function, 'kwargs': self.short_kwargs, 'status': self.status, 'attempts': self.attempts, 'queue': self.queue.queue_name, 'worker_id': self.worker_id, 'worker_name': self.worker_name}.items() if v is not None))
    return f'Job<{kwargs}>'