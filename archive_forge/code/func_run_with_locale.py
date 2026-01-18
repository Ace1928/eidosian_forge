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
def run_with_locale(catstr, *locales):

    def decorator(func):

        def inner(*args, **kwds):
            try:
                import locale
                category = getattr(locale, catstr)
                orig_locale = locale.setlocale(category)
            except AttributeError:
                raise
            except:
                locale = orig_locale = None
            else:
                for loc in locales:
                    try:
                        locale.setlocale(category, loc)
                        break
                    except:
                        pass
            try:
                return func(*args, **kwds)
            finally:
                if locale and orig_locale:
                    locale.setlocale(category, orig_locale)
        inner.__name__ = func.__name__
        inner.__doc__ = func.__doc__
        return inner
    return decorator