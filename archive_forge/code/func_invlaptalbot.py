def invlaptalbot(ctx, *args, **kwargs):
    kwargs['method'] = 'talbot'
    return ctx.invertlaplace(*args, **kwargs)