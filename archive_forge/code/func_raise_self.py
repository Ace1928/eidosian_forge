from numba import cuda
@cuda.jit(device=True)
def raise_self(x):
    if x == 1:
        raise ValueError('raise_self')
    elif x > 0:
        return raise_self(x - 1)
    else:
        return 1