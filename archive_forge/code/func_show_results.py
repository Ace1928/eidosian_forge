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
def show_results(self, stream=None):
    if stream is None:
        stream = sys.stdout
    for func, timestamps in self.functions.items():
        function_name = '%s.%s' % (func.__module__, func.__name__)
        for ts, level in zip(timestamps, self.stack[func]):
            stream.write('FUNC %s %.4f %.4f %.4f %.4f %d\n' % ((function_name,) + ts[0] + ts[1] + (level,)))