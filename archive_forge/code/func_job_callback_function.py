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
def job_callback_function(self) -> typing.Optional[typing.Callable]:
    """
        Returns the job callback function
        """
    if self.job_callback is None:
        return None
    func = import_function(self.job_callback)
    return ensure_coroutine_function(func)