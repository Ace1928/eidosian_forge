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
def test_cloned_testcase_does_not_share_details(self):
    """A TestCase cloned with clone_test does not share mutable attributes
        such as details or cleanups.
        """

    class Test(tests.TestCase):

        def test_foo(self):
            self.addDetail('foo', Content('text/plain', lambda: 'foo'))
    orig_test = Test('test_foo')
    cloned_test = tests.clone_test(orig_test, orig_test.id() + '(cloned)')
    orig_test.run(unittest.TestResult())
    self.assertEqual('foo', orig_test.getDetails()['foo'].iter_bytes())
    self.assertEqual(None, cloned_test.getDetails().get('foo'))