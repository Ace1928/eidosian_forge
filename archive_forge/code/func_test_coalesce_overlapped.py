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
def test_coalesce_overlapped(self):
    self.assertRaises(ValueError, self.check, [(0, 15, [(0, 10), (5, 10)])], [(0, 10), (5, 10)])