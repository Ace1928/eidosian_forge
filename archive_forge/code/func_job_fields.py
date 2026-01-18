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
@lazyproperty
def job_fields(self) -> typing.List[str]:
    """
        Returns the fields of the job.
        """
    return [field.name for field in self.__fields__.values()]