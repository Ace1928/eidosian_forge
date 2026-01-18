import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_generic_BzrError(self):
    self.assertTranslationEqual((b'error', b'BzrError', b'some text'), errors.BzrError(msg='some text'))