import numba as nb
def direct_gufunc_cache_usecase(**kwargs):

    @nb.guvectorize(['(intp, intp[:])', '(float64, float64[:])'], '()->()', cache=True, **kwargs)
    def gufunc(inp, out):
        out[0] = inp * 2
    return gufunc