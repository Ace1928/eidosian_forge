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
def resolve_getattr(self, typ, attr):
    """
        Resolve getting the attribute *attr* (a string) on the Numba type.
        The attribute's type is returned, or None if resolution failed.
        """

    def core(typ):
        out = self.find_matching_getattr_template(typ, attr)
        if out:
            return out['return_type']
    out = core(typ)
    if out is not None:
        return out
    out = core(types.unliteral(typ))
    if out is not None:
        return out
    if isinstance(typ, types.Module):
        attrty = self.resolve_module_constants(typ, attr)
        if attrty is not None:
            return attrty