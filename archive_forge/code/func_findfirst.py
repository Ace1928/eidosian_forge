from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def findfirst(self, py_func):
    """
        Returns the first result from `.finditer(py_func)`; or None if no match.
        """
    try:
        return next(self.finditer(py_func))
    except StopIteration:
        return