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
def run_job(self, job: 'Job') -> Job:
    """Run the job

        Args:
            job (Job): The job to run

        Returns:
            Job: _description_
        """
    job.perform()
    result_ttl = job.get_result_ttl(default_ttl=DEFAULT_RESULT_TTL)
    with self.connection.pipeline() as pipeline:
        job._handle_success(result_ttl=result_ttl, pipeline=pipeline)
        job.cleanup(result_ttl, pipeline=pipeline)
        pipeline.execute()
    return job