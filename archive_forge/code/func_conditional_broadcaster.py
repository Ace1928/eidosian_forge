import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .entry_points import load_entry_point
def conditional_broadcaster(default_func: Optional[Callable[..., Any]]=None, entry_point: Optional[str]=None) -> Callable:
    """Decorating a conditional broadcaster that will run **all** registered functions in
    other modules/packages.

    .. admonition:: Examples

        Assume in ``pkg1.module1``, you have:

        .. code-block:: python

            from triad import conditional_broadcaster

            @conditional_broadcaster(entry_point="my.plugins")
            def myprint(obj):
                raise NotImplementedError

            @conditional_broadcaster(entry_point="my.plugins")
            def myprint2(obj):
                raise NotImplementedError

        In another package ``pkg2``, in ``setup.py``, you define
        an entry point as:

        .. code-block:: python

            setup(
                ...,
                entry_points={
                    "my.plugins": [
                        "my = pkg2.module2"
                    ]
                },
            )

        And in ``pkg2.module2``:

        .. code-block:: python

            from pkg1.module1 import get_len

            @myprint.candidate(lambda obj: isinstance(obj, str))
            def myprinta(obj:str) -> None:
                print(obj, "a")

            @myprint.candidate(lambda obj: isinstance(obj, str) and obj == "x")
            def myprintb(obj:str) -> None:
                print(obj, "b")

        Now, both functions will be automatically registered when ``pkg2``
        is installed in the environement. In another ``pkg3``:

        .. code-block:: python

            from pkg1.module1 import get_len

            myprint("x")  # calling both myprinta and myprinta
            myprint("y")  # calling myprinta only
            myprint2("x")  # raise NotImplementedError due to no matching candidates

    .. note::

        Only when no matching candidate found, the implementation of the original
        function will be used. If you don't want to throw an error, then use ``pass`` in
        the original function instead.

    .. seealso::

        Please read :meth:`~.ConditionalDispatcher.candidate` for details about the
        matching function and priority settings.

    :param default_func: the function to decorate
    :param entry_point: the entry point to preload dispatchers, defaults to None
    """
    return (lambda func: ConditionalDispatcher(func, is_broadcast=True, entry_point=entry_point)) if default_func is None else ConditionalDispatcher(default_func, is_broadcast=True, entry_point=entry_point)