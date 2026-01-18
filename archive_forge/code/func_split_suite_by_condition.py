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
def split_suite_by_condition(suite, condition):
    """Split a test suite into two by a condition.

    :param suite: The suite to split.
    :param condition: The condition to match on. Tests that match this
        condition are returned in the first test suite, ones that do not match
        are in the second suite.
    :return: A tuple of two test suites, where the first contains tests from
        suite matching the condition, and the second contains the remainder
        from suite. The order within each output suite is the same as it was in
        suite.
    """
    matched = []
    did_not_match = []
    for test in iter_suite_tests(suite):
        if condition(test):
            matched.append(test)
        else:
            did_not_match.append(test)
    return (TestUtil.TestSuite(matched), TestUtil.TestSuite(did_not_match))