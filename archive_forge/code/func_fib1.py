from numba import cuda
@cuda.jit('i8(i8)', device=True)
def fib1(n):
    if n < 2:
        return n
    return fib1(n - 1) + fib1(n - 2)