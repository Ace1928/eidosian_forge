from __future__ import annotations
import functools
from lazyops.libs.pooler import ThreadPooler
from lazyops.libs.logging import logger
from typing import Any, Dict, Optional, TypeVar, Callable, TYPE_CHECKING
def get_posthog_client(**kwargs) -> 'PostHogClient':
    """
    Returns the PostHog Client
    """
    global _ph_client
    if _ph_client is None:
        from .client import PostHogClient
        _ph_client = PostHogClient(**kwargs)
    elif kwargs:
        _ph_client.configure(**kwargs)
    return _ph_client