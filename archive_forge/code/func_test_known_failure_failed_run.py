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
def test_known_failure_failed_run(self):

    class Test(tests.TestCase):

        def known_failure_test(self):
            self.expectFailure('failed', self.assertTrue, False)
    test = unittest.TestSuite()
    test.addTest(Test('known_failure_test'))

    def failing_test():
        raise AssertionError('foo')
    test.addTest(unittest.FunctionTestCase(failing_test))
    stream = StringIO()
    runner = tests.TextTestRunner(stream=stream)
    self.run_test_runner(runner, test)
    self.assertContainsRe(stream.getvalue(), "(?sm)^brz selftest.*$.*^======================================================================\n^FAIL: failing_test\n^----------------------------------------------------------------------\nTraceback \\(most recent call last\\):\n  .*    raise AssertionError\\('foo'\\)\n.*^----------------------------------------------------------------------\n.*FAILED \\(failures=1, known_failure_count=1\\)")