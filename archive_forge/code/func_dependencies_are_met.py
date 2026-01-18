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
def dependencies_are_met(self, parent_job: Optional['Job']=None, pipeline: Optional['Pipeline']=None, exclude_job_id: Optional[str]=None) -> bool:
    """Returns a boolean indicating if all of this job's dependencies are `FINISHED`

        If a pipeline is passed, all dependencies are WATCHed.

        `parent_job` allows us to directly pass parent_job for the status check.
        This is useful when enqueueing the dependents of a _successful_ job -- that status of
        `FINISHED` may not be yet set in redis, but said job is indeed _done_ and this
        method is _called_ in the _stack_ of its dependents are being enqueued.

        Args:
            parent_job (Optional[Job], optional): The parent Job. Defaults to None.
            pipeline (Optional[Pipeline], optional): The Redis' pipeline. Defaults to None.
            exclude_job_id (Optional[str], optional): Whether to exclude the job id.. Defaults to None.

        Returns:
            are_met (bool): Whether the dependencies were met.
        """
    connection = pipeline if pipeline is not None else self.connection
    if pipeline is not None:
        connection.watch(*[self.key_for(dependency_id) for dependency_id in self._dependency_ids])
    dependencies_ids = {_id.decode() for _id in connection.smembers(self.dependencies_key)}
    if exclude_job_id:
        dependencies_ids.discard(exclude_job_id)
        if parent_job and parent_job.id == exclude_job_id:
            parent_job = None
    if parent_job:
        dependencies_ids.discard(parent_job.id)
        if parent_job.get_status() == JobStatus.CANCELED:
            return False
        elif parent_job._status == JobStatus.FAILED and (not self.allow_dependency_failures):
            return False
        if not dependencies_ids:
            return True
    with connection.pipeline() as pipeline:
        for key in dependencies_ids:
            pipeline.hget(self.key_for(key), 'status')
        dependencies_statuses = pipeline.execute()
    allowed_statuses = [JobStatus.FINISHED]
    if self.allow_dependency_failures:
        allowed_statuses.append(JobStatus.FAILED)
    return all((status.decode() in allowed_statuses for status in dependencies_statuses if status))