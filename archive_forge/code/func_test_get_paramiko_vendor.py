import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_get_paramiko_vendor(self):
    """Test that if no 'ssh' is available we get builtin paramiko"""
    from breezy.transport import ssh
    self.overrideAttr(ssh, '_ssh_vendor_manager')
    self.overrideEnv('PATH', '.')
    ssh._ssh_vendor_manager.clear_cache()
    vendor = ssh._get_ssh_vendor()
    self.assertIsInstance(vendor, ssh.ParamikoVendor)