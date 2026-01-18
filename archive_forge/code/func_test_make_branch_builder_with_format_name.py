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
def test_make_branch_builder_with_format_name(self):
    builder = self.make_branch_builder('dir', format='knit')
    the_branch = builder.get_branch()
    self.assertFalse(osutils.lexists('dir'))
    dir_format = controldir.format_registry.make_controldir('knit')
    self.assertEqual(dir_format.repository_format.__class__, the_branch.repository._format.__class__)
    self.assertEqual(b'Bazaar-NG Knit Repository Format 1', self.get_transport().get_bytes('dir/.bzr/repository/format'))