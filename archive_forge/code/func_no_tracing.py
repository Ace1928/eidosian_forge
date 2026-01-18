from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def no_tracing(func):
    """Decorator to temporarily turn off tracing for the duration of a test."""
    if not hasattr(sys, 'gettrace'):
        return func
    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_trace = sys.gettrace()
            try:
                sys.settrace(None)
                return func(*args, **kwargs)
            finally:
                sys.settrace(original_trace)
        return wrapper