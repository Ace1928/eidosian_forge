import numba as nb
def direct_ufunc_cache_usecase(**kwargs):

    @nb.vectorize(['intp(intp)', 'float64(float64)'], cache=True, **kwargs)
    def ufunc(inp):
        return inp * 2
    return ufunc