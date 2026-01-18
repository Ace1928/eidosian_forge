import inspect
import logging
from inspect import Parameter
from ray._private.inspect_util import is_cython
def get_signature(func):
    """Get signature parameters.

    Support Cython functions by grabbing relevant attributes from the Cython
    function and attaching to a no-op function. This is somewhat brittle, since
    inspect may change, but given that inspect is written to a PEP, we hope
    it is relatively stable. Future versions of Python may allow overloading
    the inspect 'isfunction' and 'ismethod' functions / create ABC for Python
    functions. Until then, it appears that Cython won't do anything about
    compatability with the inspect module.

    Args:
        func: The function whose signature should be checked.

    Returns:
        A function signature object, which includes the names of the keyword
            arguments as well as their default values.

    Raises:
        TypeError: A type error if the signature is not supported
    """
    if is_cython(func):
        attrs = ['__code__', '__annotations__', '__defaults__', '__kwdefaults__']
        if all((hasattr(func, attr) for attr in attrs)):
            original_func = func

            def func():
                return
            for attr in attrs:
                setattr(func, attr, getattr(original_func, attr))
        else:
            raise TypeError(f'{func!r} is not a Python function we can process')
    return inspect.signature(func)