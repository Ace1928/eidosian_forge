import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
@staticmethod
def disable_on_exception(tqdm_instance, func):
    """
        Quietly set `tqdm_instance.miniters=inf` if `func` raises `errno=5`.
        """
    tqdm_instance = proxy(tqdm_instance)

    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno != 5:
                raise
            try:
                tqdm_instance.miniters = float('inf')
            except ReferenceError:
                pass
        except ValueError as e:
            if 'closed' not in str(e):
                raise
            try:
                tqdm_instance.miniters = float('inf')
            except ReferenceError:
                pass
    return inner