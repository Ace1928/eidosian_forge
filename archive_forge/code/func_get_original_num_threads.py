import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def get_original_num_threads(self):
    """Original num_threads from before calling threadpool_limits

        Return a dict `{user_api: num_threads}`.
        """
    num_threads = {}
    warning_apis = []
    for user_api in self._user_api:
        limits = [lib_info['num_threads'] for lib_info in self._original_info if lib_info['user_api'] == user_api]
        limits = set(limits)
        n_limits = len(limits)
        if n_limits == 1:
            limit = limits.pop()
        elif n_limits == 0:
            limit = None
        else:
            limit = min(limits)
            warning_apis.append(user_api)
        num_threads[user_api] = limit
    if warning_apis:
        warnings.warn('Multiple value possible for following user apis: ' + ', '.join(warning_apis) + '. Returning the minimum.')
    return num_threads