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
def test_allow_debug_flag(self):
    """The -Eallow_debug flag prevents breezy.debug.debug_flags from being
        sanitised (i.e. cleared) before running a test.
        """
    self.change_selftest_debug_flags({'allow_debug'})
    breezy.debug.debug_flags = {'a-flag'}

    class TestThatRecordsFlags(tests.TestCase):

        def test_foo(nested_self):
            self.flags = set(breezy.debug.debug_flags)
    test = TestThatRecordsFlags('test_foo')
    test.run(self.make_test_result())
    flags = {'a-flag'}
    if 'disable_lock_checks' not in tests.selftest_debug_flags:
        flags.add('strict_locks')
    self.assertEqual(flags, self.flags)