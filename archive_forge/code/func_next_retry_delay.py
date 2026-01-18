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
def next_retry_delay(self):
    """
        Gets the next retry delay for the job.
        """
    if self.retry_backoff:
        max_delay = self.retry_delay
        if max_delay is True:
            max_delay = None
        return exponential_backoff(attempts=self.attempts, base_delay=self.retry_delay, max_delay=max_delay, jitter=True)
    return self.retry_delay