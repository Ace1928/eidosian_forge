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
def cachify_get_exclude_keys(self, func: str, **kwargs) -> List[str]:
    """
        Gets the exclude keys
        """
    return ['background', 'callback', 'retryable', 'retry_limit', 'validate_url', 'cachable', 'disable_cache', 'overwrite_cache']