from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_get_vendor_search_order(self):
    manager = TestSSHVendorManager()
    self.overrideEnv('BRZ_SSH', None)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)
    default_vendor = object()
    manager.register_default_vendor(default_vendor)
    self.assertIs(manager.get_vendor(), default_vendor)
    manager.clear_cache()
    manager.set_ssh_version_string('OpenSSH')
    self.assertIsInstance(manager.get_vendor(), OpenSSHSubprocessVendor)
    manager.clear_cache()
    vendor = object()
    manager.register_vendor('vendor', vendor)
    self.overrideEnv('BRZ_SSH', 'vendor')
    self.assertIs(manager.get_vendor(), vendor)
    self.overrideEnv('BRZ_SSH', 'vendor')
    self.assertIs(manager.get_vendor(), vendor)