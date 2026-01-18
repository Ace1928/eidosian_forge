from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
def disable_by_count(self):
    """ Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
    if self.enable_count > 0:
        self.enable_count -= 1
        if self.enable_count == 0:
            self.disable()