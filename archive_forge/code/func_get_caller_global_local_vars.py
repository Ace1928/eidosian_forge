import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def get_caller_global_local_vars(global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None, start: int=-1, end: int=-1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get the caller level global and local variables.

    :param global_vars: overriding global variables, if not None,
      will return this instead of the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if not None,
      will return this instead of the caller's locals(), defaults to None
    :param start: start stack level (from 0 to any negative number),
      defaults to -1 which is one level above where this function is invoked
    :param end: end stack level (from ``start`` to any smaller negative number),
      defaults to -1 which is one level above where this function is invoked
    :return: tuple of `global_vars` and `local_vars`

    .. admonition:: Examples

        .. code-block:: python

            def caller():
                x=1
                assert 1 == get_value("x")

            def get_value(var_name):
                _, l = get_caller_global_local_vars()
                assert var_name in l
                assert var_name not in locals()
                return l[var_name]

    :Notice:

    This is for internal use, users normally should not call this directly.

    If merging multiple levels, the variables on closer level
    (to where it is invoked) will overwrite the further levels values if there
    is overlap.

    .. admonition:: Examples

        .. code-block:: python

            def f1():
                x=1

                def f2():
                    x=2

                    def f3():
                        _, l = get_caller_global_local_vars(start=-1,end=-2)
                        assert 2 == l["x"]

                        _, l = get_caller_global_local_vars(start=-2,end=-2)
                        assert 1 == l["x"]

                f2()
            f1()
    """
    assert_or_throw(start <= 0, ValueError(f'{start} > 0'))
    assert_or_throw(end <= start, ValueError(f'{end} > {start}'))
    stack = inspect.currentframe().f_back
    p = 0
    while p > start and stack is not None:
        stack = stack.f_back
        p -= 1
    g_arr: List[Dict[str, Any]] = []
    l_arr: List[Dict[str, Any]] = []
    while p >= end and stack is not None:
        g_arr.insert(0, stack.f_globals)
        l_arr.insert(0, stack.f_locals)
        stack = stack.f_back
        p -= 1
    if global_vars is None:
        global_vars = {}
        for d in g_arr:
            global_vars.update(d)
    if local_vars is None:
        local_vars = {}
        for d in l_arr:
            local_vars.update(d)
    return (global_vars, local_vars)