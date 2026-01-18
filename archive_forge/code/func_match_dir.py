from __future__ import absolute_import, print_function, unicode_literals
import typing
import abc
import hashlib
import itertools
import os
import six
import threading
import time
import warnings
from contextlib import closing
from functools import partial, wraps
from . import copy, errors, fsencode, iotools, tools, walk, wildcard
from .copy import copy_modified_time
from .glob import BoundGlobber
from .mode import validate_open_mode
from .path import abspath, join, normpath
from .time import datetime_to_epoch
from .walk import Walker
def match_dir(patterns, info):
    """Pattern match info.name."""
    return info.is_file or self.match(patterns, info.name)