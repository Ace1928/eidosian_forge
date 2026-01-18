import numba as nb
def direct_dufunc_cache_usecase(**kwargs):

    @nb.vectorize(cache=True, **kwargs)
    def ufunc(inp):
        return inp * 2
    return ufunc