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
def test_make_branch_builder_with_format(self):
    format = bzrdir.BzrDirMetaFormat1()
    repo_format = repository.format_registry.get_default()
    format.repository_format = repo_format
    builder = self.make_branch_builder('dir', format=format)
    the_branch = builder.get_branch()
    self.assertFalse(osutils.lexists('dir'))
    self.assertEqual(format.repository_format.__class__, the_branch.repository._format.__class__)
    self.assertEqual(repo_format.get_format_string(), self.get_transport().get_bytes('dir/.bzr/repository/format'))