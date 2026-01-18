import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
def test_jail_hook(self):
    request.jail_info.transports = None
    _pre_open_hook = request._pre_open_hook
    t = self.get_transport('foo')
    _pre_open_hook(t)
    request.jail_info.transports = [t]
    _pre_open_hook(t)
    _pre_open_hook(t.clone('child'))
    self.assertRaises(errors.JailBreak, _pre_open_hook, t.clone('..'))
    self.assertRaises(errors.JailBreak, _pre_open_hook, transport.get_transport_from_url('http://host/'))