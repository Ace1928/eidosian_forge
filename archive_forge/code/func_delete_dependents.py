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
def delete_dependents(self, pipeline: Optional['Pipeline']=None):
    """Delete jobs depending on this job.

        Args:
            pipeline (Optional[Pipeline], optional): Redis' piepline. Defaults to None.
        """
    connection = pipeline if pipeline is not None else self.connection
    for dependent_id in self.dependent_ids:
        try:
            job = Job.fetch(dependent_id, connection=self.connection, serializer=self.serializer)
            job.delete(pipeline=pipeline, remove_from_queue=False)
        except NoSuchJobError:
            pass
    connection.delete(self.dependents_key)