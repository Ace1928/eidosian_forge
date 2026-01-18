import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_generic_Exception(self):
    self.assertTranslationEqual((b'error', b'Exception', b''), Exception())