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
def test_callCatchWarnings(self):

    def meth(a, b):
        warnings.warn('this is your last warning')
        return a + b
    wlist, result = self.callCatchWarnings(meth, 1, 2)
    self.assertEqual(3, result)
    w0, = wlist
    self.assertIsInstance(w0, UserWarning)
    self.assertEqual('this is your last warning', str(w0))