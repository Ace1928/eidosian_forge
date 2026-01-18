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
def refcount_test(test):
    """Decorator for tests which involve reference counting.

    To start, the decorator does not run the test if is not run by CPython.
    After that, any trace function is unset during the test to prevent
    unexpected refcounts caused by the trace function.

    """
    return no_tracing(cpython_only(test))