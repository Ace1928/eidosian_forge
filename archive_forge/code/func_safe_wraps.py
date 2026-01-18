import functools
from inspect import iscoroutinefunction
from time import perf_counter as time_now
def safe_wraps(wrapper, *args, **kwargs):
    """Safely wraps partial functions."""
    while isinstance(wrapper, functools.partial):
        wrapper = wrapper.func
    return functools.wraps(wrapper, *args, **kwargs)