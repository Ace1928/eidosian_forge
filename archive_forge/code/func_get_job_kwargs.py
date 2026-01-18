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
def get_job_kwargs(job_data, initial_status):
    return {'func': job_data.func, 'args': job_data.args, 'kwargs': job_data.kwargs, 'result_ttl': job_data.result_ttl, 'ttl': job_data.ttl, 'failure_ttl': job_data.failure_ttl, 'description': job_data.description, 'depends_on': job_data.depends_on, 'job_id': job_data.job_id, 'meta': job_data.meta, 'status': initial_status, 'timeout': job_data.timeout, 'retry': job_data.retry, 'on_success': job_data.on_success, 'on_failure': job_data.on_failure, 'on_stopped': job_data.on_stopped}