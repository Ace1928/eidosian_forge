import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
@primitive
def container_untake(x, idx, vs):
    if isinstance(idx, slice):
        accum = lambda result: [elt_vs._mut_add(a, b) for elt_vs, a, b in zip(vs.shape[idx], result, x)]
    else:
        accum = lambda result: vs.shape[idx]._mut_add(result, x)

    def mut_add(A):
        return vs._subval(A, idx, accum(A[idx]))
    return SparseObject(vs, mut_add)