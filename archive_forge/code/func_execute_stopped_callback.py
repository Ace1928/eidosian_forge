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
def execute_stopped_callback(self, death_penalty_class: Type[BaseDeathPenalty]):
    """Executes stopped_callback with possible timeout"""
    logger.debug('Running stopped callbacks for %s', self.id)
    try:
        with death_penalty_class(self.stopped_callback_timeout, JobTimeoutException, job_id=self.id):
            self.stopped_callback(self, self.connection)
    except Exception:
        logger.exception(f'Job {self.id}: error while executing stopped callback')
        raise