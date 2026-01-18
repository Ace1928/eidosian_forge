from .functions import defun, defun_wrapped
def generalized_bisection(ctx, f, a, b, n):
    """
    Given f known to have exactly n simple roots within [a,b],
    return a list of n intervals isolating the roots
    and having opposite signs at the endpoints.

    TODO: this can be optimized, e.g. by reusing evaluation points.
    """
    if n < 1:
        raise ValueError('n cannot be less than 1')
    N = n + 1
    points = []
    signs = []
    while 1:
        points = ctx.linspace(a, b, N)
        signs = [ctx.sign(f(x)) for x in points]
        ok_intervals = [(points[i], points[i + 1]) for i in range(N - 1) if signs[i] * signs[i + 1] == -1]
        if len(ok_intervals) == n:
            return ok_intervals
        N = N * 2