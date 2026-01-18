import os
from functools import wraps
def py3_data(init_func):

    def _decorator(*args, **kwargs):
        args = (args[0], add_py3_data(args[1])) + args[2:]
        return init_func(*args, **kwargs)
    return wraps(init_func)(_decorator)