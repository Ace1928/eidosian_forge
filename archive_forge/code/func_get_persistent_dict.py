from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def get_persistent_dict(self, base_key: str, expiration: Optional[int]=None, aliases: Optional[List[str]]=None, **kwargs) -> 'PersistentDict':
    """
        Lazily initializes a persistent dict
        """
    if base_key not in self._persistent_dicts and base_key not in self._persistent_dict_aliases:
        url = kwargs.pop('url', None)
        session = self.get_kdb_session('persistence', serializer=None, url=url)
        self._persistent_dicts[base_key] = session.create_persistence(base_key=base_key, expiration=expiration, **kwargs)
        if aliases:
            for alias in aliases:
                self._persistent_dict_aliases[alias] = base_key
    elif base_key in self._persistent_dict_aliases:
        base_key = self._persistent_dict_aliases[base_key]
    return self._persistent_dicts[base_key]