from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_default_vendor(self):
    manager = TestSSHVendorManager()
    self.overrideEnv('BRZ_SSH', None)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)
    vendor = object()
    manager.register_default_vendor(vendor)
    self.assertIs(manager.get_vendor(), vendor)