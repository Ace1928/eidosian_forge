import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
def get_config_copy(self):
    return copy.deepcopy(self._config)