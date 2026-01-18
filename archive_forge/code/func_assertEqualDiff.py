import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def assertEqualDiff(self, a, b, message=None):
    """Assert two texts are equal, if not raise an exception.

        This is intended for use with multi-line strings where it can
        be hard to find the differences by eye.
        """
    if a == b:
        return
    if message is None:
        message = 'texts not equal:\n'
    if a + ('\n' if isinstance(a, str) else b'\n') == b:
        message = 'first string is missing a final newline.\n'
    if a == b + ('\n' if isinstance(b, str) else b'\n'):
        message = 'second string is missing a final newline.\n'
    raise AssertionError(message + self._ndiff_strings(a if isinstance(a, str) else a.decode(), b if isinstance(b, str) else b.decode()))