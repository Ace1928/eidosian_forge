from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def select_overload(self, sig, overloads, allow_unsafe, exact_match_required):
    sig = [t._code for t in sig]
    overloads = [[t._code for t in s] for s in overloads]
    return _typeconv.select_overload(self._ptr, sig, overloads, allow_unsafe, exact_match_required)