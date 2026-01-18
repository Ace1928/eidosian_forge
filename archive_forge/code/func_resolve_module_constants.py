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
def resolve_module_constants(self, typ, attr):
    """
        Resolve module-level global constants.
        Return None or the attribute type
        """
    assert isinstance(typ, types.Module)
    attrval = getattr(typ.pymod, attr)
    try:
        return self.resolve_value_type(attrval)
    except ValueError:
        pass