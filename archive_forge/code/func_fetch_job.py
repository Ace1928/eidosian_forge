import logging
import sys
import traceback
import uuid
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from redis import WatchError
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_RESULT_TTL
from .dependency import Dependency
from .exceptions import DequeueTimeout, NoSuchJobError
from .job import Callback, Job, JobStatus
from .logutils import blue, green
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import as_text, backend_class, compact, get_version, import_attribute, parse_timeout, utcnow
def fetch_job(self, job_id: str) -> Optional['Job']:
    """Fetch a single job by Job ID.
        If the job key is not found, will run the `remove` method, to exclude the key.
        If the job has the same name as as the current job origin, returns the Job

        Args:
            job_id (str): The Job ID

        Returns:
            job (Optional[Job]): The job if found
        """
    try:
        job = self.job_class.fetch(job_id, connection=self.connection, serializer=self.serializer)
    except NoSuchJobError:
        self.remove(job_id)
    else:
        if job.origin == self.name:
            return job