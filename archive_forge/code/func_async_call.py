import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def async_call(target: Callable, timeout: Optional[Union[int, float]]=None) -> Callable:
    """Wrap a method to run in the background with an optional timeout.

    Returns a new method that will call the original with any args, waiting for upto
    timeout seconds. This new method blocks on the original and returns the result or
    None if timeout was reached, along with the thread. You can check thread.is_alive()
    to determine if a timeout was reached. If an exception is thrown in the thread, we
    reraise it.
    """
    q: queue.Queue = queue.Queue()

    def wrapped_target(q: 'queue.Queue', *args: Any, **kwargs: Any) -> Any:
        try:
            q.put(target(*args, **kwargs))
        except Exception as e:
            q.put(e)

    def wrapper(*args: Any, **kwargs: Any) -> Union[Tuple[Exception, 'threading.Thread'], Tuple[None, 'threading.Thread']]:
        thread = threading.Thread(target=wrapped_target, args=(q,) + args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        try:
            result = q.get(True, timeout)
            if isinstance(result, Exception):
                raise result.with_traceback(sys.exc_info()[2])
            return (result, thread)
        except queue.Empty:
            return (None, thread)
    return wrapper