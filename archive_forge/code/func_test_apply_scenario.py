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
def test_apply_scenario(self):
    from breezy.tests import apply_scenario
    input_test = TestTestScenarioApplication('test_apply_scenario')
    adapted_test1 = apply_scenario(input_test, ('new id', {'bzrdir_format': 'bzr_format', 'repository_format': 'repo_fmt', 'transport_server': 'transport_server', 'transport_readonly_server': 'readonly-server'}))
    adapted_test2 = apply_scenario(input_test, ('new id 2', {'bzrdir_format': None}))
    self.assertRaises(AttributeError, getattr, input_test, 'bzrdir_format')
    self.assertEqual('bzr_format', adapted_test1.bzrdir_format)
    self.assertEqual('repo_fmt', adapted_test1.repository_format)
    self.assertEqual('transport_server', adapted_test1.transport_server)
    self.assertEqual('readonly-server', adapted_test1.transport_readonly_server)
    self.assertEqual('breezy.tests.test_selftest.TestTestScenarioApplication.test_apply_scenario(new id)', adapted_test1.id())
    self.assertEqual(None, adapted_test2.bzrdir_format)
    self.assertEqual('breezy.tests.test_selftest.TestTestScenarioApplication.test_apply_scenario(new id 2)', adapted_test2.id())