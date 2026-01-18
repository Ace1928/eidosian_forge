from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger
from lazyops.utils.lazy import lazy_import
from lazyops.utils.helpers import fail_after
from typing import Any, Callable, Dict, List, Optional, Union, Type
def get_az_kdb(name: Optional[str]=None, serializer: Optional[str]='json', **kwargs) -> 'KVDBSession':
    """
    Returns the KVDB Session
    """
    global _az_kdbs
    if name is None:
        name = 'global'
    if name not in _az_kdbs:
        _az_kdbs[name] = kvdb.KVDBClient.get_session(name=name, serializer=serializer, **kwargs)
    return _az_kdbs[name]