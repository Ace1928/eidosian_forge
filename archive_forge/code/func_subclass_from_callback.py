from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
@classmethod
@deprecated(use_instead='MassAction.from_callback')
def subclass_from_callback(cls, cb, cls_attrs=None):
    """Override MassAction.__call__"""
    _RateExpr = super(MassAction, cls).subclass_from_callback(cb, cls_attrs=cls_attrs)

    def wrapper(*args, **kwargs):
        obj = _RateExpr(*args, **kwargs)
        return cls(obj)
    return wrapper