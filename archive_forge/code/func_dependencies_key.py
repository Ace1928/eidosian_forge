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
@property
def dependencies_key(self):
    return '{0}:{1}:dependencies'.format(self.redis_job_namespace_prefix, self.id)