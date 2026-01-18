from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def create_new_class(code: str, cls: DynamicT, ref: Optional[str]=None, module_name: Optional[str]=None) -> DynamicT:
    """
    Create a new class
    """
    global _DynamicClasses
    module_name = module_name or 'lazyops.types.dynamic'
    from .serialization import create_hash_key
    code_hash = create_hash_key(kwargs={'ref': ref, 'cls': cls, 'module_name': module_name, 'code': code})
    if code_hash in _DynamicClasses:
        return _DynamicClasses[code_hash]
    run_code = import_code(code, add_to_sys=True, ref=ref or cls.__name__, module_name=module_name)
    new_cls = copy.deepcopy(cls)
    for prop_or_func in dir(run_code):
        setattr(new_cls, prop_or_func, getattr(run_code, prop_or_func))
    _DynamicClasses[code_hash] = new_cls
    return new_cls