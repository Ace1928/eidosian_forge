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
def stuck(self):
    """
        Checks if an active job is passed it's timeout or heartbeat.
        - if timeout is None, set timeout to 2 hrs = 7200.00
        - revised timeout to 30 mins = 1800.00
        """
    current = now()
    return self.status == JobStatus.ACTIVE and (seconds(current - self.started) > (self.timeout if self.timeout is not None else 7200.0) or (self.heartbeat and seconds(current - self.touched) > self.heartbeat))