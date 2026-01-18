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
def test_coalesce_max_size(self):
    self.check([(10, 20, [(0, 10), (10, 10)]), (30, 50, [(0, 50)]), (100, 80, [(0, 80)])], [(10, 10), (20, 10), (30, 50), (100, 80)], max_size=50)