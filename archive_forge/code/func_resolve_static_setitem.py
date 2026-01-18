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
def resolve_static_setitem(self, target, index, value):
    assert not isinstance(index, types.Type), index
    args = (target, index, value)
    kws = {}
    return self.resolve_function_type('static_setitem', args, kws)