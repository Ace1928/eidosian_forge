import os
import sys
import asyncio
import threading
from uuid import uuid4
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from lazyops.models import LazyData
from lazyops.common import lazy_import, lazylibs
from lazyops.retry import retryable
def require_module(name, *args, **kwargs):

    def decorator(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            submod = lazylibs.get_submodule(name)
            if not submod:
                submod = lazylibs.setup_submodule(name, *args, **kwargs)
            if not submod.has_initialized:
                submod.lazy_init()
            return f(*args, **kwargs)
        return wrapper
    return decorator