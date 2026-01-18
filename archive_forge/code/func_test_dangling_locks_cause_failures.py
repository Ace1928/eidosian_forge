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
def test_dangling_locks_cause_failures(self):

    class TestDanglingLock(tests.TestCaseWithMemoryTransport):

        def test_function(self):
            t = self.get_transport_from_path('.')
            l = lockdir.LockDir(t, 'lock')
            l.create()
            l.attempt_lock()
    test = TestDanglingLock('test_function')
    result = test.run()
    total_failures = result.errors + result.failures
    if self._lock_check_thorough:
        self.assertEqual(1, len(total_failures))
    else:
        self.assertEqual(0, len(total_failures))