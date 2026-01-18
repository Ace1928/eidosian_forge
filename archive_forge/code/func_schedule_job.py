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
def schedule_job(self, job: 'Job', datetime: datetime, pipeline: Optional['Pipeline']=None):
    """Puts job on ScheduledJobRegistry

        Args:
            job (Job): _description_
            datetime (datetime): _description_
            pipeline (Optional[Pipeline], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
    from .registry import ScheduledJobRegistry
    registry = ScheduledJobRegistry(queue=self)
    pipe = pipeline if pipeline is not None else self.connection.pipeline()
    pipe.sadd(self.redis_queues_keys, self.key)
    job.save(pipeline=pipe)
    registry.schedule(job, datetime, pipeline=pipe)
    if pipeline is None:
        pipe.execute()
    return job