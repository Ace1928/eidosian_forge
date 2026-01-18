import atexit
import os
import platform
import random
import sys
import threading
import time
import uuid
from collections import deque
import sentry_sdk
from sentry_sdk._compat import PY33, PY311
from sentry_sdk._lru_cache import LRUCache
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def has_profiling_enabled(options):
    profiles_sampler = options['profiles_sampler']
    if profiles_sampler is not None:
        return True
    profiles_sample_rate = options['profiles_sample_rate']
    if profiles_sample_rate is not None and profiles_sample_rate > 0:
        return True
    profiles_sample_rate = options['_experiments'].get('profiles_sample_rate')
    if profiles_sample_rate is not None and profiles_sample_rate > 0:
        return True
    return False