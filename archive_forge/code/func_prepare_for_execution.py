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
def prepare_for_execution(self, worker_name: str, pipeline: 'Pipeline'):
    """Prepares the job for execution, setting the worker name,
        heartbeat information, status and other metadata before execution begins.

        Args:
            worker_name (str): The worker that will perform the job
            pipeline (Pipeline): The Redis' piipeline to use
        """
    self.worker_name = worker_name
    self.last_heartbeat = utcnow()
    self.started_at = self.last_heartbeat
    self._status = JobStatus.STARTED
    mapping = {'last_heartbeat': utcformat(self.last_heartbeat), 'status': self._status, 'started_at': utcformat(self.started_at), 'worker_name': worker_name}
    if self.get_redis_server_version() >= (4, 0, 0):
        pipeline.hset(self.key, mapping=mapping)
    else:
        pipeline.hmset(self.key, mapping=mapping)