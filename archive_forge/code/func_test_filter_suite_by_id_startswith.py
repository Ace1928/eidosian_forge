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
def test_filter_suite_by_id_startswith(self):
    klass = 'breezy.tests.test_selftest.TestSelftestFiltering.'
    start1 = klass + 'test_filter_suite_by_id_starts'
    start2 = klass + 'test_filter_suite_by_id_li'
    test_list = [klass + 'test_filter_suite_by_id_list', klass + 'test_filter_suite_by_id_startswith']
    filtered_suite = tests.filter_suite_by_id_startswith(self.suite, [start1, start2])
    self.assertEqual(test_list, _test_ids(filtered_suite))