from numba import cuda
@cuda.jit('i8(i8)', device=True)
def fib2(n):
    if n < 2:
        return n
    return fib2(n - 1) + fib2(n - 2)