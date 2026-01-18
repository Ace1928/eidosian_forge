from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
def log_hook(fn: Callable, level=logging.INFO) -> Callable:
    """
    Logs callable output.

    This is useful for logging output of passes. Note inplace_wrapper replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before inplace_wrapper.


    ```
    def my_pass(d: Dict) -> bool:
        changed = False
        if 'foo' in d:
            d['foo'] = 'bar'
            changed = True
        return changed

    pm = PassManager(
        passes=[
            inplace_wrapper(log_hook(my_pass))
        ]
    )
    ```

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        val = fn(gm)
        logger.log(level, 'Ran pass %s\t Return value: %s', fn, val)
        return val
    return wrapped_fn