from __future__ import annotations
import ast
import sys
import types
from collections.abc import Callable, Iterable
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from os import PathLike
from types import CodeType, ModuleType, TracebackType
from typing import Sequence, TypeVar
from unittest.mock import patch
from ._config import global_config
from ._transformer import TypeguardTransformer
def should_instrument(self, module_name: str) -> bool:
    """
        Determine whether the module with the given name should be instrumented.

        :param module_name: full name of the module that is about to be imported (e.g.
            ``xyz.abc``)

        """
    if self.packages is None:
        return True
    for package in self.packages:
        if module_name == package or module_name.startswith(package + '.'):
            return True
    return False