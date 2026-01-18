from __future__ import annotations
import inspect
from abc import ABC
from urllib.parse import urljoin
from fastapi import HTTPException
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.utils.helpers import timed_cache
from ..types.errors import InvalidOperationException
from ..types.auth import AuthZeroTokenAuth
from ..types.clients import AuthZeroClientObject
from ..utils.lazy import get_az_settings, logger
from .tokens import ClientCredentialsFlow
from typing import Optional, List, Dict, Any, Union
@timed_cache(600)
def get_service_client_name(self, client_id: str) -> Optional[str]:
    """
        Returns the Service Client Name
        """
    client = self.get_az_client(client_id=client_id)
    return None if client is None else client.name