import inspect
import sys
def echo_module(mod, write=sys.stdout.write):
    """ Echo calls to functions and methods in a module.
    """
    for fname, fn in inspect.getmembers(mod, inspect.isfunction):
        setattr(mod, fname, echo(fn, write))
    for _, klass in inspect.getmembers(mod, inspect.isclass):
        echo_class(klass, write)