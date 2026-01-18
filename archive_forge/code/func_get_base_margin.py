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
def get_base_margin(self) -> np.ndarray:
    """Get the base margin of the DMatrix.

        Returns
        -------
        base_margin
        """
    return self.get_float_info('base_margin')