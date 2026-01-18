import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_local_parameters(self):
    t = self.get_nfs_transport('.')
    self.assertEqual(True, t.listable())
    self.assertEqual(False, t.is_readonly())