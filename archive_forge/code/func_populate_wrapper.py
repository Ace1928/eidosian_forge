import functools
import inspect
from textwrap import dedent
def populate_wrapper(klass, wrapping):
    for meth, how in klass._wrap_methods.items():
        if not hasattr(wrapping, meth):
            continue
        func = getattr(wrapping, meth)
        wrapper = make_wrapper(func, how)
        setattr(klass, meth, wrapper)