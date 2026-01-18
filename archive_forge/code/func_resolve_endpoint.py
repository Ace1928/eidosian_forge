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
def resolve_endpoint(endpoints: List[str]) -> str:
    """
    Resolves the endpoint
    """
    for endpoint in endpoints:
        ep = tldextract.extract(endpoint)
        with contextlib.suppress(Exception):
            if ep.registered_domain:
                socket.gethostbyname(ep.registered_domain)
            else:
                socket.gethostbyaddr(ep.ipv4)
            return endpoint
    raise ValueError(f'Could not resolve endpoint: {endpoints}')