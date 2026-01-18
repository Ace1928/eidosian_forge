from numba import jit
@jit(nopython=True)
def runaway_mutual_inner(x):
    return runaway_mutual(x)