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
@contextlib.contextmanager
def swap_attr(obj, attr, new_val):
    """Temporary swap out an attribute with a new object.

    Usage:
        with swap_attr(obj, "attr", 5):
            ...

        This will set obj.attr to 5 for the duration of the with: block,
        restoring the old value at the end of the block. If `attr` doesn't
        exist on `obj`, it will be created and then deleted at the end of the
        block.
    """
    if hasattr(obj, attr):
        real_val = getattr(obj, attr)
        setattr(obj, attr, new_val)
        try:
            yield
        finally:
            setattr(obj, attr, real_val)
    else:
        setattr(obj, attr, new_val)
        try:
            yield
        finally:
            delattr(obj, attr)