import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def func_matches(stat, funcs):
    """
    This function will not work with stats that are saved and loaded. That is 
    because current API of loading stats is as following:
    yappi.get_func_stats(filter_callback=_filter).add('dummy.ys').print_all()

    funcs: is an iterable that selects functions via method descriptor/bound method
        or function object. selector type depends on the function object: If function
        is a builtin method, you can use method_descriptor. If it is a builtin function
        you can select it like e.g: `time.sleep`. For other cases you could use anything 
        that has a code object.
    """
    if not isinstance(stat, YStat):
        raise YappiError(f"Argument 'stat' shall be a YStat object. ({stat})")
    if not isinstance(funcs, list):
        raise YappiError(f"Argument 'funcs' is not a list object. ({funcs})")
    if not len(funcs):
        raise YappiError("Argument 'funcs' cannot be empty.")
    if stat.full_name not in _fn_descriptor_dict:
        return False
    funcs = set(funcs)
    for func in funcs.copy():
        if not callable(func):
            raise YappiError(f"Non-callable item in 'funcs'. ({func})")
        if getattr(func, '__code__', None):
            funcs.add(func.__code__)
    try:
        return _fn_descriptor_dict[stat.full_name] in funcs
    except TypeError:
        return False