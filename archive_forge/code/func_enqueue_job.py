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
def enqueue_job(self, job: 'Job', pipeline: Optional['Pipeline']=None, at_front: bool=False) -> Job:
    """Enqueues a job for delayed execution checking dependencies.

        Args:
            job (Job): The job to enqueue
            pipeline (Optional[Pipeline], optional): The Redis pipeline to use. Defaults to None.
            at_front (bool, optional): Whether should enqueue at the front of the queue. Defaults to False.

        Returns:
            Job: The enqued job
        """
    job.origin = self.name
    job = self.setup_dependencies(job, pipeline=pipeline)
    if job.get_status(refresh=False) != JobStatus.DEFERRED:
        return self._enqueue_job(job, pipeline=pipeline, at_front=at_front)
    return job