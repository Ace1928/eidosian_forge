from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
@property
def temp_data(self) -> 'TemporaryData':
    """
        Retrieves the temporary data
        """
    if self._temp_data is None:
        from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
        self._temp_data = TemporaryData.from_module(self.module_name)
    return self._temp_data