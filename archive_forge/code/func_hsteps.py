from ..libmp.backend import xrange
from .calculus import defun
def hsteps(ctx, f, x, n, prec, **options):
    singular = options.get('singular')
    addprec = options.get('addprec', 10)
    direction = options.get('direction', 0)
    workprec = (prec + 2 * addprec) * (n + 1)
    orig = ctx.prec
    try:
        ctx.prec = workprec
        h = options.get('h')
        if h is None:
            if options.get('relative'):
                hextramag = int(ctx.mag(x))
            else:
                hextramag = 0
            h = ctx.ldexp(1, -prec - addprec - hextramag)
        else:
            h = ctx.convert(h)
        direction = options.get('direction', 0)
        if direction:
            h *= ctx.sign(direction)
            steps = xrange(n + 1)
            norm = h
        else:
            steps = xrange(-n, n + 1, 2)
            norm = 2 * h
        if singular:
            x += 0.5 * h
        values = [f(x + k * h) for k in steps]
        return (values, norm, workprec)
    finally:
        ctx.prec = orig