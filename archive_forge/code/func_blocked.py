import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
@contextlib.contextmanager
def blocked(self, *, signal=None):
    """
        Block callback signals from being processed.

        A context manager to temporarily block/disable callback signals
        from being processed by the registered listeners.

        Parameters
        ----------
        signal : str, optional
            The callback signal to block. The default is to block all signals.
        """
    orig = self.callbacks
    try:
        if signal is None:
            self.callbacks = {}
        else:
            self.callbacks = {k: orig[k] for k in orig if k != signal}
        yield
    finally:
        self.callbacks = orig