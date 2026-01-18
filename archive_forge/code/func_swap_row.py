from ..libmp.backend import xrange
import warnings
def swap_row(ctx, A, i, j):
    """
        Swap row i with row j.
        """
    if i == j:
        return
    if isinstance(A, ctx.matrix):
        for k in xrange(A.cols):
            A[i, k], A[j, k] = (A[j, k], A[i, k])
    elif isinstance(A, list):
        A[i], A[j] = (A[j], A[i])
    else:
        raise TypeError('could not interpret type')