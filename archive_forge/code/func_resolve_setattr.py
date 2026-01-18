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
def resolve_setattr(self, target, attr, value):
    """
        Resolve setting the attribute *attr* (a string) on the *target* type
        to the given *value* type.
        A function signature is returned, or None if resolution failed.
        """
    for attrinfo in self._get_attribute_templates(target):
        expectedty = attrinfo.resolve(target, attr)
        if expectedty is not None:
            return templates.signature(types.void, target, expectedty)