import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def clear_jail_info():
    request.jail_info.transports = None