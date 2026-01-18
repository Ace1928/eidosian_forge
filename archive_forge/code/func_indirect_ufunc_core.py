import numba as nb
@nb.njit(cache=True)
def indirect_ufunc_core(inp):
    return inp * 3