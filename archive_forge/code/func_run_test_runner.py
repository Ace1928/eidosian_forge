import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def run_test_runner(self, testrunner, test):
    """Run suite in testrunner, saving global state and restoring it.

        This current saves and restores:
        TestCaseInTempDir.TEST_ROOT

        There should be no tests in this file that use
        breezy.tests.TextTestRunner without using this convenience method,
        because of our use of global state.
        """
    old_root = tests.TestCaseInTempDir.TEST_ROOT
    try:
        tests.TestCaseInTempDir.TEST_ROOT = None
        return testrunner.run(test)
    finally:
        tests.TestCaseInTempDir.TEST_ROOT = old_root