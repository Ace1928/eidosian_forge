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
def hpatch(self, path: str, **kwargs) -> 'Response':
    """
        Returns the Response
        """
    if not self.use_http_session:
        return niquests.patch(self.get_url(path), auth=self.auth, **kwargs)
    return self.session.patch(self.get_url(path), auth=self.auth, **kwargs)