from .decorators import jit
import numba
@jit(device=True)
def shfl_sync(mask, value, src_lane):
    """
    Shuffles value across the masked warp and returns the value
    from src_lane. If this is outside the warp, then the
    given value is returned.
    """
    return numba.cuda.shfl_sync_intrinsic(mask, 0, value, src_lane, 31)[0]