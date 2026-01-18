from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def quadraticstring(ctx, t, a, b, c):
    if c < 0:
        a, b, c = (-a, -b, -c)
    u1 = (-b + ctx.sqrt(b ** 2 - 4 * a * c)) / (2 * c)
    u2 = (-b - ctx.sqrt(b ** 2 - 4 * a * c)) / (2 * c)
    if abs(u1 - t) < abs(u2 - t):
        if b:
            s = '((%s+sqrt(%s))/%s)' % (-b, b ** 2 - 4 * a * c, 2 * c)
        else:
            s = '(sqrt(%s)/%s)' % (-4 * a * c, 2 * c)
    elif b:
        s = '((%s-sqrt(%s))/%s)' % (-b, b ** 2 - 4 * a * c, 2 * c)
    else:
        s = '(-sqrt(%s)/%s)' % (-4 * a * c, 2 * c)
    return s