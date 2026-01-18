from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
def init_ctx(self, module_name: str, *args, **kwargs) -> ApplicationContext:
    """
        Initializes the app context
        """
    if module_name not in self.ctxs:
        self.ctxs[module_name] = ApplicationContext(module_name, *args, global_ctx=self.global_ctx, **kwargs)
    return self.ctxs[module_name]