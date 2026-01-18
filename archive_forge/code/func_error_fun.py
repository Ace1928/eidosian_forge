from numba import jit
@jit
def error_fun(x):
    return x.ndim