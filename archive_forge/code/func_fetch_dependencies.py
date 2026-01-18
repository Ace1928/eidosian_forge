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
def fetch_dependencies(self, watch: bool=False, pipeline: Optional['Pipeline']=None) -> List['Job']:
    """Fetch all of a job's dependencies. If a pipeline is supplied, and
        watch is true, then set WATCH on all the keys of all dependencies.

        Returned jobs will use self's connection, not the pipeline supplied.

        If a job has been deleted from redis, it is not returned.

        Args:
            watch (bool, optional): Wether to WATCH the keys. Defaults to False.
            pipeline (Optional[Pipeline]): The Redis' pipeline to use. Defaults to None.

        Returns:
            jobs (list[Job]): A list of Jobs
        """
    connection = pipeline if pipeline is not None else self.connection
    if watch and self._dependency_ids:
        connection.watch(*[self.key_for(dependency_id) for dependency_id in self._dependency_ids])
    dependencies_list = self.fetch_many(self._dependency_ids, connection=self.connection, serializer=self.serializer)
    jobs = [job for job in dependencies_list if job]
    return jobs