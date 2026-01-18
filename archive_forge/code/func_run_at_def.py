import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
def run_at_def(run_at_def_func: Optional[Callable]=None, **kwargs: Any) -> Callable:
    """Decorator to run the function at declaration. This is useful when we want import
    to trigger a function run (which can guarantee it runs only once).

    .. admonition:: Examples

        Assume the following python file is a module in your package,
        then when you ``import package.module``, the two functions will run.

        .. code-block:: python

            from triad import run_at_def

            @run_at_def
            def register_something():
                print("registered")

            @run_at_def(a=1)
            def register_something2(a):
                print("registered", a)

    :param run_at_def_func: the function to decorate
    :param kwargs: the parameters to call this function
    """

    def _run(_func: Callable) -> Callable:
        _func(**kwargs)
        return _func
    return _run if run_at_def_func is None else _run(run_at_def_func)