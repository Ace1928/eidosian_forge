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
def get_jobs_to_schedule(self, timestamp: Optional[datetime]=None, chunk_size: int=1000) -> List[str]:
    """Get's a list of job IDs that should be scheduled.

        Args:
            timestamp (Optional[datetime], optional): _description_. Defaults to None.
            chunk_size (int, optional): _description_. Defaults to 1000.

        Returns:
            jobs (List[str]): A list of Job ids
        """
    score = timestamp if timestamp is not None else current_timestamp()
    jobs_to_schedule = self.connection.zrangebyscore(self.key, 0, score, start=0, num=chunk_size)
    return [as_text(job_id) for job_id in jobs_to_schedule]