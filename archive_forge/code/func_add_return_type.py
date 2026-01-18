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
def add_return_type(self, return_type):
    """Add *return_type* to the list of inferred return-types.
        If there are too many, raise `TypingError`.
        """
    RETTY_LIMIT = 16
    self._inferred_retty.add(return_type)
    if len(self._inferred_retty) >= RETTY_LIMIT:
        m = 'Return type of recursive function does not converge'
        raise errors.TypingError(m)