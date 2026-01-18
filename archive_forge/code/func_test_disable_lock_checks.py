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
def test_disable_lock_checks(self):
    """The -Edisable_lock_checks flag disables thorough checks."""

    class TestThatRecordsFlags(tests.TestCase):

        def test_foo(nested_self):
            self.flags = set(breezy.debug.debug_flags)
            self.test_lock_check_thorough = nested_self._lock_check_thorough
    self.change_selftest_debug_flags(set())
    test = TestThatRecordsFlags('test_foo')
    test.run(self.make_test_result())
    self.assertTrue(self.test_lock_check_thorough)
    self.assertEqual({'strict_locks'}, self.flags)
    self.change_selftest_debug_flags({'disable_lock_checks'})
    test = TestThatRecordsFlags('test_foo')
    test.run(self.make_test_result())
    self.assertFalse(self.test_lock_check_thorough)
    self.assertEqual(set(), self.flags)