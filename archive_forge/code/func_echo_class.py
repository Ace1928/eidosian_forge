import inspect
import sys
def echo_class(klass, write=sys.stdout.write):
    """ Echo calls to class methods and static functions
    """
    for _, method in inspect.getmembers(klass, inspect.ismethod):
        echo_instancemethod(klass, method, write)
    for _, fn in inspect.getmembers(klass, inspect.isfunction):
        if is_static_method(fn, klass):
            setattr(klass, name(fn), staticmethod(echo(fn, write)))
        else:
            echo_instancemethod(klass, fn, write)