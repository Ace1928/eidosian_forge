import calendar
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union
from rq.serializers import resolve_serializer
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_FAILURE_TTL
from .exceptions import AbandonedJobError, InvalidJobOperation, NoSuchJobError
from .job import Job, JobStatus
from .queue import Queue
from .utils import as_text, backend_class, current_timestamp
def get_scheduled_time(self, job_or_id: Union['Job', str]) -> datetime:
    """Returns datetime (UTC) at which job is scheduled to be enqueued

        Args:
            job_or_id (Union[Job, str]): The Job instance or Job ID

        Raises:
            NoSuchJobError: If the job was not found

        Returns:
            datetime (datetime): The scheduled time as datetime object
        """
    if isinstance(job_or_id, self.job_class):
        job_id = job_or_id.id
    else:
        job_id = job_or_id
    score = self.connection.zscore(self.key, job_id)
    if not score:
        raise NoSuchJobError
    return datetime.fromtimestamp(score, tz=timezone.utc)