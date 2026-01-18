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
def callDeprecated(self, expected, callable, *args, **kwargs):
    """Assert that a callable is deprecated in a particular way.

        This is a very precise test for unusual requirements. The
        applyDeprecated helper function is probably more suited for most tests
        as it allows you to simply specify the deprecation format being used
        and will ensure that that is issued for the function being called.

        Note that this only captures warnings raised by symbol_versioning.warn,
        not other callers that go direct to the warning module.  To catch
        general warnings, use callCatchWarnings.

        :param expected: a list of the deprecation warnings expected, in order
        :param callable: The callable to call
        :param args: The positional arguments for the callable
        :param kwargs: The keyword arguments for the callable
        """
    call_warnings, result = self._capture_deprecation_warnings(callable, *args, **kwargs)
    self.assertEqual(expected, call_warnings)
    return result