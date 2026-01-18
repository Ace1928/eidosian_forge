from __future__ import annotations
import functools
from lazyops.libs.pooler import ThreadPooler
from lazyops.libs.logging import logger
from typing import Any, Dict, Optional, TypeVar, Callable, TYPE_CHECKING
def has_existing_posthog_client() -> bool:
    """
    Checks if there is an existing PostHog Client
    """
    global _ph_client
    return _ph_client is not None