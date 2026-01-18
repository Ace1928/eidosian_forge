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
def test_coalesce_default_limit(self):
    ten_mb = 10 * 1024 * 1024
    self.check([(0, 10 * ten_mb, [(i * ten_mb, ten_mb) for i in range(10)]), (10 * ten_mb, ten_mb, [(0, ten_mb)])], [(i * ten_mb, ten_mb) for i in range(11)])
    self.check([(0, 11 * ten_mb, [(i * ten_mb, ten_mb) for i in range(11)])], [(i * ten_mb, ten_mb) for i in range(11)], max_size=1 * 1024 * 1024 * 1024)