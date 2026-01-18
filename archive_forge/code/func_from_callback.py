import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
@classmethod
def from_callback(cls, callback, attr='__call__', **kwargs):
    Wrapper = super().from_callback(callback, attr=attr, **kwargs)
    return lambda *args, **kw: cls(Wrapper(*args, **kw))