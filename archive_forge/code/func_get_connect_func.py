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
def get_connect_func(self, _is_async: bool=False):
    if self.async_redis_connect_func and _is_async:
        return import_string(self.async_redis_connect_func)
    elif self.redis_connect_func and (not _is_async):
        return import_string(self.redis_connect_func)
    return None