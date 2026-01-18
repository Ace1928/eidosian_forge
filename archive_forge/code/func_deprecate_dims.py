import inspect
import warnings
from functools import wraps
from typing import Callable, TypeVar
from xarray.core.utils import emit_user_level_warning
def deprecate_dims(func: T) -> T:
    """
    For functions that previously took `dims` as a kwarg, and have now transitioned to
    `dim`. This decorator will issue a warning if `dims` is passed while forwarding it
    to `dim`.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'dims' in kwargs:
            emit_user_level_warning('The `dims` argument has been renamed to `dim`, and will be removed in the future. This renaming is taking place throughout xarray over the next few releases.', PendingDeprecationWarning)
            kwargs['dim'] = kwargs.pop('dims')
        return func(*args, **kwargs)
    return wrapper