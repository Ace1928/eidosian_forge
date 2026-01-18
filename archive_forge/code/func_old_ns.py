from collections import deque, namedtuple
from datetime import timedelta
from celery.utils.functional import memoize
from celery.utils.serialization import strtobool
def old_ns(ns):
    return {f'{ns}_{{0}}'}