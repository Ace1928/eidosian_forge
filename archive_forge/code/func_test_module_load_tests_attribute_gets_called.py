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
def test_module_load_tests_attribute_gets_called(self):
    loader, module = self._get_loader_and_module()

    def load_tests(loader, standard_tests, pattern):
        result = loader.suiteClass()
        for test in tests.iter_suite_tests(standard_tests):
            result.addTests([test, test])
        return result
    module.__class__.load_tests = staticmethod(load_tests)
    self.assertEqual(2 * [str(module.a_class('test_foo'))], list(map(str, loader.loadTestsFromModule(module))))