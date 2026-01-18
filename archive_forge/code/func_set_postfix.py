import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
    """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
    postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
    for key in sorted(kwargs.keys()):
        postfix[key] = kwargs[key]
    for key in postfix.keys():
        if isinstance(postfix[key], Number):
            postfix[key] = self.format_num(postfix[key])
        elif not isinstance(postfix[key], str):
            postfix[key] = str(postfix[key])
    self.postfix = ', '.join((key + '=' + postfix[key].strip() for key in postfix.keys()))
    if refresh:
        self.refresh()