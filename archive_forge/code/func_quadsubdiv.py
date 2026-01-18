import math
from ..libmp.backend import xrange
def quadsubdiv(ctx, f, interval, tol=None, maxintervals=None, **kwargs):
    """
        Computes the integral of *f* over the interval or path specified
        by *interval*, using :func:`~mpmath.quad` together with adaptive
        subdivision of the interval.

        This function gives an accurate answer for some integrals where
        :func:`~mpmath.quad` fails::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> quad(lambda x: abs(sin(x)), [0, 2*pi])
            3.99900894176779
            >>> quadsubdiv(lambda x: abs(sin(x)), [0, 2*pi])
            4.0
            >>> quadsubdiv(sin, [0, 1000])
            0.437620923709297
            >>> quadsubdiv(lambda x: 1/(1+x**2), [-100, 100])
            3.12159332021646
            >>> quadsubdiv(lambda x: ceil(x), [0, 100])
            5050.0
            >>> quadsubdiv(lambda x: sin(x+exp(x)), [0,8])
            0.347400172657248

        The argument *maxintervals* can be set to limit the permissible
        subdivision::

            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=5, error=True)
            (-5.40487904307774, 5.011)
            >>> quadsubdiv(lambda x: sin(x**2), [0,100], maxintervals=100, error=True)
            (0.631417921866934, 1.10101120134116e-17)

        Subdivision does not guarantee a correct answer since, the error
        estimate on subintervals may be inaccurate::

            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.210802735500549, 1.0001111101e-17)
            >>> mp.dps = 20
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, [0,1], error=True)
            (0.21080273550054927738, 2.200000001e-24)

        The second answer is correct. We can get an accurate result at lower
        precision by forcing a finer initial subdivision::

            >>> mp.dps = 15
            >>> quadsubdiv(lambda x: sech(10*x-2)**2 + sech(100*x-40)**4 + sech(1000*x-600)**6, linspace(0,1,5))
            0.210802735500549

        The following integral is too oscillatory for convergence, but we can get a
        reasonable estimate::

            >>> v, err = fp.quadsubdiv(lambda x: fp.sin(1/x), [0,1], error=True)
            >>> round(v, 6), round(err, 6)
            (0.504067, 1e-06)
            >>> sin(1) - ci(1)
            0.504067061906928

        """
    queue = []
    for i in range(len(interval) - 1):
        queue.append((interval[i], interval[i + 1]))
    total = ctx.zero
    total_error = ctx.zero
    if maxintervals is None:
        maxintervals = 10 * ctx.prec
    count = 0
    quad_args = kwargs.copy()
    quad_args['verbose'] = False
    quad_args['error'] = True
    if tol is None:
        tol = +ctx.eps
    orig = ctx.prec
    try:
        ctx.prec += 5
        while queue:
            a, b = queue.pop()
            s, err = ctx.quad(f, [a, b], **quad_args)
            if kwargs.get('verbose'):
                print('subinterval', count, a, b, err)
            if err < tol or count > maxintervals:
                total += s
                total_error += err
            else:
                count += 1
                if count == maxintervals and kwargs.get('verbose'):
                    print('warning: number of intervals exceeded maxintervals')
                if a == -ctx.inf and b == ctx.inf:
                    m = 0
                elif a == -ctx.inf:
                    m = min(b - 1, 2 * b)
                elif b == ctx.inf:
                    m = max(a + 1, 2 * a)
                else:
                    m = a + (b - a) / 2
                queue.append((a, m))
                queue.append((m, b))
    finally:
        ctx.prec = orig
    if kwargs.get('error'):
        return (+total, +total_error)
    else:
        return +total