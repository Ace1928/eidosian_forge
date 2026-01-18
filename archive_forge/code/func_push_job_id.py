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
def push_job_id(self, job_id: str, pipeline: Optional['Pipeline']=None, at_front: bool=False):
    """Pushes a job ID on the corresponding Redis queue.
        'at_front' allows you to push the job onto the front instead of the back of the queue

        Args:
            job_id (str): The Job ID
            pipeline (Optional[Pipeline], optional): The Redis Pipeline to use. Defaults to None.
            at_front (bool, optional): Whether to push the job to front of the queue. Defaults to False.
        """
    connection = pipeline if pipeline is not None else self.connection
    push = connection.lpush if at_front else connection.rpush
    result = push(self.key, job_id)
    if pipeline is None:
        self.log.debug('Pushed job %s into %s, %s job(s) are in queue.', blue(job_id), green(self.name), result)
    else:
        self.log.debug('Pushed job %s into %s', blue(job_id), green(self.name))