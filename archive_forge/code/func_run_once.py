from functools import _make_key, wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple, Type
def run_once(func: Optional[Callable]=None, key_func: Optional[Callable]=None, lock_type: Type=RLock) -> Callable:
    """The decorator to run `func` once, the uniqueness is defined by `key_func`.
    This implementation is serialization safe and thread safe.

    :param func: the function to run only once with this wrapper instance
    :param key_func: the unique key determined by arguments of `func`, if not set, it
      will use the same hasing logic as :external+python:func:`functools.lru_cache`
    :param lock_type: lock class type for thread safe, it doesn't need to be
      serialization safe

    .. admonition:: Examples

        .. code-block:: python

            @run_once
            def r(a):
                return max(a)

            a1 = [0, 1]
            a2 = [0, 2]
            assert 1 == r(a1) # will trigger r
            assert 1 == r(a1) # will get the result from cache
            assert 2 == r(a2) # will trigger r again because of different arguments

            # the following example ignores arguments
            @run_once(key_func=lambda *args, **kwargs: True)
            def r2(a):
                return max(a)

            assert 1 == r(a1) # will trigger r
            assert 1 == r(a2) # will get the result from cache

    .. note::

        * Hash collision is the concern of the user, not this class, your
          `key_func` should avoid any potential collision
        * `func` can have no return
        * For concurrent calls of this wrapper, only one will trigger `func` other
          calls will be blocked until the first call returns an result
        * This class is cloudpicklable, but unpickled instance does NOT share the same
          context with the original one
        * This is not to replace :external+python:func:`functools.lru_cache`,
          it is not supposed to cache a lot of items
    """

    def _run(func: Callable) -> 'RunOnce':
        return wraps(func)(RunOnce(func, key_func=key_func, lock_type=lock_type))
    return _run(func) if func is not None else wraps(func)(_run)