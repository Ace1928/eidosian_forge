import contextlib
import copy
import hashlib
import inspect
import io
import pickle
import tokenize
import unittest
import warnings
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Tuple, Union
from unittest import mock
def shallow_copy_dict(self) -> Dict[str, Any]:
    return {**self._config}