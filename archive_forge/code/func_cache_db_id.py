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
@property
def cache_db_id(self) -> int:
    """
        returns the `cache` DB ID
        - if it's found in ``self.db_mapping``
        - otherwise it will set to the next db id
        """
    if self.db_mapping and self.db_mapping.get('cache') is not None:
        return self.db_mapping['cache']
    db_id = (len(self.db_mapping) if self.db_mapping else 1) + 1
    if not self.db_mapping:
        self.db_mapping = {}
    self.db_mapping['cache'] = db_id
    return db_id