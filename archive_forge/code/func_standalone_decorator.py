import functools
def standalone_decorator(f):

    def standalone_wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return standalone_wrapper