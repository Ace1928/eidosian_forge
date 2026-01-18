from __future__ import annotations
from typing import Dict, TypeVar, Optional, Type, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.utils.logs import logger
def register_client(client: Union[ClientT, ClientTypeT], name: str, verbose: Optional[bool]=False, **kwargs) -> bool:
    """
    Registers a client
    """
    global _registered_clients
    if name not in _registered_clients:
        _registered_clients[name] = client
        if verbose:
            logger.info(f'Client {name}', colored=True, prefix='|g|Registered|e|')
        return True
    return False