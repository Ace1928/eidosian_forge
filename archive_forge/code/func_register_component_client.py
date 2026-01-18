from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def register_component_client(self, client: 'ClientTypes', *parts: str, kind: Optional[str]=None, include_kind: Optional[bool]=None):
    """
        Registers a component client
        """
    from lazyops.libs.fastapi_utils.state.registry import register_client
    include_kind = include_kind if include_kind is not None else self.include_kind_in_component_name
    kind = kind or getattr(client, 'kind', None)
    if include_kind:
        prefix = f'{self.module_name}.{kind}' if kind else self.module_name
    else:
        prefix = self.module_name
    if parts:
        parts = '.'.join(parts)
        prefix = f'{prefix}.{parts}'
    client_name = f'{prefix}.{client.name}'
    return register_client(client, client_name)