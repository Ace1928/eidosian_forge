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
def disable_enforce_env(self) -> bool:
    """
        Returns whether to disable the enforce env
        """
    return os.getenv('DISABLE_ENFORCE_ENV', 'false').lower() in {'true', '1'}