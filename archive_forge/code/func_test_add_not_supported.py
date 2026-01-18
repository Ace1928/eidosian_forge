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
def test_add_not_supported(self):
    """Test the behaviour of invoking addNotSupported."""

    class InstrumentedTestResult(tests.ExtendedTestResult):

        def stopTestRun(self):
            pass

        def report_tests_starting(self):
            pass

        def report_unsupported(self, test, feature):
            self._call = (test, feature)
    result = InstrumentedTestResult(None, None, None, None)
    test = SampleTestCase('_test_pass')
    feature = features.Feature()
    result.startTest(test)
    result.addNotSupported(test, feature)
    self.assertEqual(2, len(result._call))
    self.assertEqual(test, result._call[0])
    self.assertEqual(feature, result._call[1])
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(1, result.unsupported['Feature'])
    result.addNotSupported(test, feature)
    self.assertEqual(2, result.unsupported['Feature'])