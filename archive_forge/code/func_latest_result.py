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
def latest_result(self, timeout: int=0) -> Optional['Result']:
    """Get the latest job result.

        Args:
            timeout (int, optional): Number of seconds to block waiting for a result. Defaults to 0 (no blocking).

        Returns:
            result (Result): The Result object
        """
    from .results import Result
    return Result.fetch_latest(self, serializer=self.serializer, timeout=timeout)