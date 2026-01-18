import numba as nb
def indirect_ufunc_cache_usecase(**kwargs):

    @nb.njit(cache=True)
    def indirect_ufunc_core(inp):
        return inp * 3

    @nb.vectorize(['intp(intp)', 'float64(float64)', 'complex64(complex64)'], **kwargs)
    def ufunc(inp):
        return indirect_ufunc_core(inp)
    return ufunc