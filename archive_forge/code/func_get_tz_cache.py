import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
@classmethod
def get_tz_cache(cls):
    if cls._tz_cache is None:
        with _cache_init_lock:
            cls._initialise()
    return cls._tz_cache