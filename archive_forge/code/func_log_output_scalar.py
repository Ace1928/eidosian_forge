def log_output_scalar(name, data, run=None):
    if isinstance_namedtuple(data):
        for k, v in zip(data._fields, data):
            run.log({f'{func.__name__}.{k}': v})
    else:
        run.log({name: data})