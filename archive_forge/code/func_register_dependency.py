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
def register_dependency(self, pipeline: Optional['Pipeline']=None):
    """Jobs may have dependencies. Jobs are enqueued only if the jobs they
        depend on are successfully performed. We record this relation as
        a reverse dependency (a Redis set), with a key that looks something
        like:
        ..codeblock:python::

            rq:job:job_id:dependents = {'job_id_1', 'job_id_2'}

        This method adds the job in its dependencies' dependents sets,
        and adds the job to DeferredJobRegistry.

        Args:
            pipeline (Optional[Pipeline]): The Redis' pipeline. Defaults to None
        """
    from .registry import DeferredJobRegistry
    registry = DeferredJobRegistry(self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer)
    registry.add(self, pipeline=pipeline)
    connection = pipeline if pipeline is not None else self.connection
    for dependency_id in self._dependency_ids:
        dependents_key = self.dependents_key_for(dependency_id)
        connection.sadd(dependents_key, self.id)
        connection.sadd(self.dependencies_key, dependency_id)