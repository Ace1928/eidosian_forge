from numba import vectorize
@vectorize(['int64(int64, int64)'], nopython=True)
def inner_explicit(a, b):
    return a + b