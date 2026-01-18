from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
def get_server_domain(request: Optional['Request']=None, module_domains: Optional[List[str]]=None, module_name: Optional[str]=None, verbose: Optional[bool]=True, force_https: Optional[bool]=None) -> Optional[str]:
    """
    Get the server domain that the app is hosted on
    """
    global _server_domains
    if not _server_domains.get(module_name):
        if request is None:
            return None
        if not module_domains:
            module_domains = ['localhost', module_name]
        if any((domain in request.url.hostname for domain in module_domains)):
            scheme = 'https' if force_https or request.url.port == 443 else request.url.scheme
            _server_domains[module_name] = f'{scheme}://{request.url.hostname}'
            if request.url.port and request.url.port not in {80, 443}:
                _server_domains[module_name] += f':{request.url.port}'
            if verbose:
                logger.info(f'[|g|{module_name}|e|] Setting Server Domain: {_server_domains[module_name]} from {request.url}', colored=True)
    return _server_domains.get(module_name)