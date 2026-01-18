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
def test_condition_id_in_list(self):
    test_names = ['breezy.tests.test_selftest.TestSelftestFiltering.test_condition_id_in_list']
    id_list = tests.TestIdList(test_names)
    filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_in_list(id_list))
    my_pattern = 'TestSelftestFiltering.*test_condition_id_in_list'
    re_filtered = tests.filter_suite_by_re(self.suite, my_pattern)
    self.assertEqual(_test_ids(re_filtered), _test_ids(filtered_suite))