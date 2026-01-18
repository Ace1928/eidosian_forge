from __future__ import annotations
import os
import json
import socket
import contextlib
import logging
from typing import Optional, Dict, Any, Union, Type, Mapping, Callable, List
from lazyops.utils.logs import default_logger as logger
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.types import BaseSettings, validator, root_validator, lazyproperty, KeyDBUri
from aiokeydb.v2.serializers import SerializerType, BaseSerializer
from aiokeydb.v2.utils import import_string
from aiokeydb.v2.configs.worker import KeyDBWorkerSettings
from aiokeydb.v2.backoff import default_backoff
def get_retry_arg(self, _is_async: bool=False) -> Dict[str, Any]:
    """
        Returns the retry argument
        """
    if _is_async:
        from redis.asyncio.retry import Retry
    else:
        from redis.retry import Retry
    return {'retry': Retry(default_backoff(), retries=3, supported_errors=(exceptions.ConnectionError, exceptions.TimeoutError, exceptions.BusyLoadingError))} if self.retry_enabled else {}