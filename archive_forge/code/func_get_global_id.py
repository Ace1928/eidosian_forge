import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def get_global_id(builder, dim):
    sreg = SRegBuilder(builder)
    it = (sreg.getdim(xyz) for xyz in 'xyz')
    seq = list(itertools.islice(it, None, dim))
    if dim == 1:
        return seq[0]
    else:
        return seq