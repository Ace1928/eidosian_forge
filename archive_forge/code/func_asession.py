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
@property
def asession(self) -> 'AsyncSession':
    """
        Returns the Async Session
        """
    if self._asession is None:
        self._asession = niquests.AsyncSession(**self.get_session_kwargs(is_async=True))
    return self._asession