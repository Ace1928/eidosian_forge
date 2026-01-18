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
def preflight_check(self):
    """
        Conducts a pre-flight check
        """
    if not self.available:
        raise ValueError(f'API is not available: {self.endpoint}')
    self.check_authorization()