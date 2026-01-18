from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger
from lazyops.utils.lazy import lazy_import
from lazyops.utils.helpers import fail_after
from typing import Any, Callable, Dict, List, Optional, Union, Type
def get_az_mtg_api() -> 'AZManagementAPI':
    """
    Returns the AZ Management API
    """
    global _az_mtg_api
    if _az_mtg_api is None:
        from ..flows.admin import AZManagementClient
        _az_mtg_api = AZManagementClient
    return _az_mtg_api