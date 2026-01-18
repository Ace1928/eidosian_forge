from .functions import defun, defun_wrapped
def f_wrapped(ctx):
    cache = ctx._misc_const_cache
    prec = ctx.prec
    p, v = cache.get(name, (-1, 0))
    if p >= prec:
        return +v
    else:
        cache[name] = (prec, f(ctx))
        return cache[name][1]