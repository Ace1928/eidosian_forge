from pyomo.common.plugin_base import (
def pyomo_callback(name):
    """This is a decorator that declares a function to be
    a callback function.  The callback functions are
    added to the solver when run from the pyomo script.

    Example:

    @pyomo_callback('cut-callback')
    def my_cut_generator(solver, model):
        ...
    """

    def fn(f):
        registered_callback[name] = f
        return f
    return fn