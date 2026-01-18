from numba import cuda
def make_growing_tuple_case(jit=lambda x: x):

    @jit
    def make_list(n):
        if n <= 0:
            return None
        return (n, make_list(n - 1))
    return make_list