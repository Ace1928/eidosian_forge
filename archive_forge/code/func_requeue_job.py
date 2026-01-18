import asyncio
import inspect
import json
import logging
import warnings
import zlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from redis import WatchError
from .defaults import CALLBACK_TIMEOUT, UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .timeouts import BaseDeathPenalty, JobTimeoutException
from .connections import resolve_connection
from .exceptions import DeserializationError, InvalidJobOperation, NoSuchJobError
from .local import LocalStack
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import (
def requeue_job(job_id: str, connection: 'Redis', serializer=None) -> 'Job':
    """Fetches a Job by ID and requeues it using the `requeue()` method.

    Args:
        job_id (str): The Job ID that should be requeued.
        connection (Redis): The Redis Connection to use
        serializer (Optional[str], optional): The serializer. Defaults to None.

    Returns:
        Job: The requeued Job object.
    """
    job = Job.fetch(job_id, connection=connection, serializer=serializer)
    return job.requeue()