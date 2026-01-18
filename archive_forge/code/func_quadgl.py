import math
from ..libmp.backend import xrange
def quadgl(ctx, *args, **kwargs):
    """
        Performs Gauss-Legendre quadrature. The call

            quadgl(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=GaussLegendre)

        For example, a single integral and a double integral:

            quadgl(lambda x: exp(cos(x)), [0, 1])
            quadgl(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
    kwargs['method'] = 'gauss-legendre'
    return ctx.quad(*args, **kwargs)