import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
@contextlib.contextmanager
def override_temp(replacement):
    """
    Monkey-patch tempfile.tempdir with replacement, ensuring it exists
    """
    os.makedirs(replacement, exist_ok=True)
    saved = tempfile.tempdir
    tempfile.tempdir = replacement
    try:
        yield
    finally:
        tempfile.tempdir = saved