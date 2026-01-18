import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def outside_wrapper(function: CallableT) -> CallableT:

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return function(*args, **kwargs)
    return cast(CallableT, wrapper)