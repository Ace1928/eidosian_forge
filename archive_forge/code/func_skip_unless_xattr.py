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
def skip_unless_xattr(test):
    """Skip decorator for tests that require functional extended attributes"""
    ok = can_xattr()
    msg = 'no non-broken extended attribute support'
    return test if ok else unittest.skip(msg)(test)