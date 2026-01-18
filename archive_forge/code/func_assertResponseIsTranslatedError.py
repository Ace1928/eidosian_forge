import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def assertResponseIsTranslatedError(self, handler):
    expected_translation = (b'NoSuchFile', b'xyzzy')
    self.assertEqual(request.FailedSmartServerResponse(expected_translation), handler.response)