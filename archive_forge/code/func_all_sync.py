from .decorators import jit
import numba
@jit(device=True)
def all_sync(mask, predicate):
    """
    If for all threads in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 0, predicate)[1]