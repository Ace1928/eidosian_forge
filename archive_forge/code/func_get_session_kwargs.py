from __future__ import annotations
import socket
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from lazyops.utils.helpers import timed_cache
from lazyops.libs.abcs.configs.types import AppEnv
from ..utils.lazy import get_az_settings, logger, get_az_flow, get_az_resource
from typing import Optional, List, Dict, Any, Union
def get_session_kwargs(self, is_async: Optional[bool]=None, **kwargs) -> Dict[str, Any]:
    """
        Returns the session kwargs
        """
    return {'pool_connections': self._kwargs.get('pool_connections', 10), 'pool_maxsize': self._kwargs.get('pool_maxsize', 10), 'retries': self._kwargs.get('retries', self.default_retries), **kwargs}