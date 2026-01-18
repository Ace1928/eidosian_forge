from __future__ import annotations
import asyncio
import random
import inspect
import aiohttpx
import functools
import subprocess
from pydantic import BaseModel
from urllib.parse import urlparse
from lazyops.libs.proxyobj import ProxyObject, proxied
from .base import BaseGlobalClient, cachify
from .utils import aget_root_domain, get_user_agent, http_retry_wrapper
from typing import Optional, Type, TypeVar, Literal, Union, Set, Awaitable, Any, Dict, List, Callable, overload, TYPE_CHECKING
@property
def google_csx_api_key(self) -> str:
    """
        Override this to add a Google CSX API Key
        """
    if self._google_csx_api_key is None:
        if hasattr(self.settings, 'clients') and hasattr(self.settings.clients, 'http_pool'):
            self._google_csx_api_key = getattr(self.settings.clients.http_pool, 'google_csx_api_key', None)
    return self._google_csx_api_key