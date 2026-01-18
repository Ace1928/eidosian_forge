import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_disabled_method(self):
    error = request.DisabledMethod('class name')
    self.assertEqualDiff("The smart server method 'class name' is disabled.", str(error))