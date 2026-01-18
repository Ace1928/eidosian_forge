def f_monitored(*args, **kwargs):
    input((args, kwargs))
    v = f(*args, **kwargs)
    output(v)
    return v