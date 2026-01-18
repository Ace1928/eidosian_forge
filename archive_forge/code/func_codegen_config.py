import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
def codegen_config(self):
    """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
    lines = []
    mod = self.__name__
    for k, v in self._config.items():
        if k in self._config.get('_save_config_ignore', ()):
            continue
        if v == self._default[k]:
            continue
        lines.append(f'{mod}.{k} = {v!r}')
    return '\n'.join(lines)