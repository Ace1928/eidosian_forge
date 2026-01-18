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
@cachify.register()
def get_pdftotext(self, url: str, cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> Optional[str]:
    """
        Transform a PDF File to Text directly from URL
        """
    return self._get_pdftotext(url, retryable=retryable, retry_limit=retry_limit, **kwargs)