from ..libmp.backend import xrange
from .calculus import defun
def fold_infinite(ctx, f, intervals):
    if len(intervals) < 2:
        return f
    dim1 = intervals[-2][0]
    dim2 = intervals[-1][0]

    def g(*args):
        args = list(args)
        n = int(args[dim1])
        s = ctx.zero
        args[dim2] = ctx.mpf(n)
        for x in xrange(n + 1):
            args[dim1] = ctx.mpf(x)
            s += f(*args)
        args[dim1] = ctx.mpf(n)
        for y in xrange(n):
            args[dim2] = ctx.mpf(y)
            s += f(*args)
        return s
    return fold_infinite(ctx, g, intervals[:-1])