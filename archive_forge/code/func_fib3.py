from numba import cuda
@cuda.jit(device=True)
def fib3(n):
    if n < 2:
        return n
    return fib3(n - 1) + fib3(n - 2)