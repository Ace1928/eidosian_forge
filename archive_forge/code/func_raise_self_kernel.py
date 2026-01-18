from numba import cuda
@cuda.jit(debug=True, opt=False)
def raise_self_kernel(x):
    raise_self(x)