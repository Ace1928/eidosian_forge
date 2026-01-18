import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_generic_zlib_error(self):
    from zlib import error
    msg = 'Error -3 while decompressing data: incorrect data check'
    self.assertTranslationEqual((b'error', b'zlib.error', msg.encode('utf-8')), error(msg))