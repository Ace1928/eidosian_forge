from ..libmp.backend import xrange
def eps_pade(p):
    return ctx.mpf(2) ** (3 - 2 * p) * ctx.factorial(p) ** 2 / (ctx.factorial(2 * p) ** 2 * (2 * p + 1))