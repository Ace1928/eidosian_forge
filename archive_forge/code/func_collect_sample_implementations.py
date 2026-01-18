import importlib
import inspect
import pkgutil
import sys
from types import ModuleType
from typing import Dict
def collect_sample_implementations() -> Dict[str, str]:
    dict_: Dict[str, str] = {}
    _recursive_scan(sys.modules[__name__], dict_)
    return dict_