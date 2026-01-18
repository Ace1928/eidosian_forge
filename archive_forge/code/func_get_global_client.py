from __future__ import annotations
from typing import Dict, TypeVar, Optional, Type, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.utils.logs import logger
def get_global_client(name: str) -> ClientTypeT:
    """
    Global Clients do not require initialization and are types
    """
    global _registered_clients
    if not _client_registry_mapping:
        raise ValueError('Client Registry Mapping not set')
    if name not in _registered_clients:
        if name not in _client_registry_mapping:
            raise ValueError(f'Client {name} not found in client registry mapping')
        _registered_clients[name] = lazy_import(_client_registry_mapping[name])
    return _registered_clients[name]