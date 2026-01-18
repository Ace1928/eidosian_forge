import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def num_boosted_rounds(self) -> int:
    """Get number of boosted rounds.  For gblinear this is reset to 0 after
        serializing the model.

        """
    rounds = ctypes.c_int()
    assert self.handle is not None
    _check_call(_LIB.XGBoosterBoostedRounds(self.handle, ctypes.byref(rounds)))
    return rounds.value