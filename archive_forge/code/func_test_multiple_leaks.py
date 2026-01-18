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
def test_multiple_leaks(self):
    """Check multiple leaks are blamed on the test cases at fault

        Same concept as the previous test, but has one inner test method that
        leaks two threads, and one that doesn't leak at all.
        """
    event = threading.Event()
    thread_a = threading.Thread(name='LeakerA', target=event.wait)
    thread_b = threading.Thread(name='LeakerB', target=event.wait)
    thread_c = threading.Thread(name='LeakerC', target=event.wait)

    class Test(tests.TestCase):

        def test_first_leak(self):
            thread_b.start()

        def test_second_no_leak(self):
            pass

        def test_third_leak(self):
            thread_c.start()
            thread_a.start()
    result = self.LeakRecordingResult()
    first_test = Test('test_first_leak')
    third_test = Test('test_third_leak')
    self.addCleanup(thread_a.join)
    self.addCleanup(thread_b.join)
    self.addCleanup(thread_c.join)
    self.addCleanup(event.set)
    result.startTestRun()
    unittest.TestSuite([first_test, Test('test_second_no_leak'), third_test]).run(result)
    result.stopTestRun()
    self.assertEqual(result._tests_leaking_threads_count, 2)
    self.assertEqual(result._first_thread_leaker_id, first_test.id())
    self.assertEqual(result.leaks, [(first_test, {thread_b}), (third_test, {thread_a, thread_c})])
    self.assertContainsString(result.stream.getvalue(), 'leaking threads')