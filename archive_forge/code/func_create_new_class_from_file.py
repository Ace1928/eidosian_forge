from __future__ import annotations
import sys
import copy
import pathlib
import inspect
import functools
import importlib.util
from typing import Any, Dict, Callable, Union, Optional, Type, TypeVar, List, Tuple, cast, TYPE_CHECKING
from types import ModuleType
def create_new_class_from_file(file: Union[str, pathlib.Path], cls_name: str, cls: DynamicT) -> DynamicT:
    """
    Import a file and create a new class
    """
    module_spec = load_module_from_file(file=file, module_name=cls_name)
    cls_spec = getattr(module_spec, cls_name)
    cls_spec = cast(cls, cls_spec)
    return cls_spec