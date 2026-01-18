import sys
def get_qualified_name(function):
    if hasattr(function, '__qualname__'):
        return function.__qualname__
    if hasattr(function, 'im_class'):
        return function.im_class.__name__ + '.' + function.__name__
    return function.__name__