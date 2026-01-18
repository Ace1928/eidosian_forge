from functools import update_wrapper, wraps
from types import MethodType
def available_if(check):
    """An attribute that is available only if check returns a truthy value.

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Returns
    -------
    callable
        Callable makes the decorated method available if `check` returns
        a truthy value, otherwise the decorated method is unavailable.

    Examples
    --------
    >>> from sklearn.utils.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)