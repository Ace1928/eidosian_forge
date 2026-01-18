from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def failed_dispatch(loader, queue, error):
    """
    Do not cache individual loads if the entire batch dispatch fails,
    but still reject each request so they do not hang.
    """
    for l in queue:
        loader.clear(l.key)
        l.reject(error)