def invlapstehfest(ctx, *args, **kwargs):
    kwargs['method'] = 'stehfest'
    return ctx.invertlaplace(*args, **kwargs)