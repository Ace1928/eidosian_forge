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
def resolve_value_type_prefer_literal(self, value):
    """Resolve value type and prefer Literal types whenever possible.
        """
    lit = types.maybe_literal(value)
    if lit is None:
        return self.resolve_value_type(value)
    else:
        return lit