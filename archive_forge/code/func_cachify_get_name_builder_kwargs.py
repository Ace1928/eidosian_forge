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
def cachify_get_name_builder_kwargs(self, func: str, **kwargs) -> Dict[str, Any]:
    """
        Gets the name builder kwargs
        """
    return {'include_http_methods': True, 'special_names': ['pdftotext', 'csx']}