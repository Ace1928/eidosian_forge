import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
def proper_eq(a: Any, b: Any) -> bool:
    """Compares objects for equality, working around __eq__ not always working.

    For example, in numpy a == b broadcasts and returns an array instead of
    doing what np.array_equal(a, b) does. This method uses np.array_equal(a, b)
    when dealing with numpy arrays.
    """
    if type(a) == type(b):
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, (pd.DataFrame, pd.Index, pd.MultiIndex)):
            return a.equals(b)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all((proper_eq(x, y) for x, y in zip(a, b)))
    return a == b