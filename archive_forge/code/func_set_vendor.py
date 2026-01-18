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
def set_vendor(self, vendor, subprocess_stderr=None):
    from breezy.transport import ssh
    self.overrideAttr(ssh._ssh_vendor_manager, '_cached_ssh_vendor', vendor)
    if subprocess_stderr is not None:
        self.overrideAttr(ssh.SubprocessVendor, '_stderr_target', subprocess_stderr)