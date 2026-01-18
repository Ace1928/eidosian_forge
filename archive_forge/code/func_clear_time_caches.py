import time
from functools import wraps
from typing import Any, Dict, Tuple
from jedi import settings
from parso.cache import parser_cache
def clear_time_caches(delete_all: bool=False) -> None:
    """ Jedi caches many things, that should be completed after each completion
    finishes.

    :param delete_all: Deletes also the cache that is normally not deleted,
        like parser cache, which is important for faster parsing.
    """
    global _time_caches
    if delete_all:
        for cache in _time_caches.values():
            cache.clear()
        parser_cache.clear()
    else:
        for tc in _time_caches.values():
            for key, (t, value) in list(tc.items()):
                if t < time.time():
                    del tc[key]