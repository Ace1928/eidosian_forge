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
def test_known_failure(self):
    """Using knownFailure should trigger several result actions."""

    class InstrumentedTestResult(tests.ExtendedTestResult):

        def stopTestRun(self):
            pass

        def report_tests_starting(self):
            pass

        def report_known_failure(self, test, err=None, details=None):
            self._call = (test, 'known failure')
    result = InstrumentedTestResult(None, None, None, None)

    class Test(tests.TestCase):

        def test_function(self):
            self.knownFailure('failed!')
    test = Test('test_function')
    test.run(result)
    self.assertEqual(2, len(result._call))
    self.assertEqual(test.id(), result._call[0].id())
    self.assertEqual('known failure', result._call[1])
    self.assertEqual(1, result.known_failure_count)
    self.assertTrue(result.wasSuccessful())