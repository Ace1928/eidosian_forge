from functools import partial, update_wrapper, wraps
from asgiref.sync import iscoroutinefunction
def sync_and_async_middleware(func):
    """
    Mark a middleware factory as returning a hybrid middleware supporting both
    types of request.
    """
    func.sync_capable = True
    func.async_capable = True
    return func