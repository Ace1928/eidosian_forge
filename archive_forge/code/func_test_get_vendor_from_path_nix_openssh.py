from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_get_vendor_from_path_nix_openssh(self):
    manager = TestSSHVendorManager()
    manager.set_ssh_version_string('OpenSSH_5.1p1 Debian-5, OpenSSL, 0.9.8g 19 Oct 2007')
    openssh_path = '/usr/bin/ssh'
    self.overrideEnv('BRZ_SSH', openssh_path)
    vendor = manager.get_vendor()
    self.assertIsInstance(vendor, OpenSSHSubprocessVendor)
    args = vendor._get_vendor_specific_argv('user', 'host', 22, ['bzr'])
    self.assertEqual(args[0], openssh_path)