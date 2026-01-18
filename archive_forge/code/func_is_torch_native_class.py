import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def is_torch_native_class(cls):
    if not hasattr(cls, '__module__'):
        return False
    parent_modules = cls.__module__.split('.')
    if not parent_modules:
        return False
    root_module = sys.modules.get(parent_modules[0])
    return root_module is torch