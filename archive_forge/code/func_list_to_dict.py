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
def list_to_dict(items, delim='='):
    res = {}
    for item in items:
        i = item.split(delim, 1)
        res[i[0].strip()] = i[-1].strip()
    return res