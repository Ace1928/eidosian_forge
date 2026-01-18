import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
def export_case(**kwargs):
    """
    Decorator for registering a user provided case into example bank.
    """

    def wrapper(m):
        configs = kwargs
        module = inspect.getmodule(m)
        if module in _MODULES:
            raise RuntimeError('export_case should only be used once per example file.')
        _MODULES.add(module)
        normalized_name = to_snake_case(m.__name__)
        assert module is not None
        module_name = module.__name__.split('.')[-1]
        if module_name != normalized_name:
            raise RuntimeError(f'Module name "{module.__name__}" is inconsistent with exported program ' + f'name "{m.__name__}". Please rename the module to "{normalized_name}".')
        case = _make_export_case(m, module_name, configs)
        register_db_case(case)
        return case
    return wrapper