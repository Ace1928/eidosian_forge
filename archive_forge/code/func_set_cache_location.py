import peewee as _peewee
from threading import Lock
import os as _os
import appdirs as _ad
import atexit as _atexit
import datetime as _datetime
import pickle as _pkl
from .utils import get_yf_logger
def set_cache_location(cache_dir: str):
    """
    Sets the path to create the "py-yfinance" cache folder in.
    Useful if the default folder returned by "appdir.user_cache_dir()" is not writable.
    Must be called before cache is used (that is, before fetching tickers).
    :param cache_dir: Path to use for caches
    :return: None
    """
    _TzDBManager.set_location(cache_dir)
    _CookieDBManager.set_location(cache_dir)