import numba as nb
def indirect_gufunc_cache_usecase(**kwargs):

    @nb.njit(cache=True)
    def core(x):
        return x * 3

    @nb.guvectorize(['(intp, intp[:])', '(float64, float64[:])', '(complex64, complex64[:])'], '()->()', **kwargs)
    def gufunc(inp, out):
        out[0] = core(inp)
    return gufunc