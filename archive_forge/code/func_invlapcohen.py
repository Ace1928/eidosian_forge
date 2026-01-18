def invlapcohen(ctx, *args, **kwargs):
    kwargs['method'] = 'cohen'
    return ctx.invertlaplace(*args, **kwargs)