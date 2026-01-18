import math
from ..libmp.backend import xrange
def quadts(ctx, *args, **kwargs):
    """
        Performs tanh-sinh quadrature. The call

            quadts(func, *points, ...)

        is simply a shortcut for:

            quad(func, *points, ..., method=TanhSinh)

        For example, a single integral and a double integral:

            quadts(lambda x: exp(cos(x)), [0, 1])
            quadts(lambda x, y: exp(cos(x+y)), [0, 1], [0, 1])

        See the documentation for quad for information about how points
        arguments and keyword arguments are parsed.

        See documentation for TanhSinh for algorithmic information about
        tanh-sinh quadrature.
        """
    kwargs['method'] = 'tanh-sinh'
    return ctx.quad(*args, **kwargs)