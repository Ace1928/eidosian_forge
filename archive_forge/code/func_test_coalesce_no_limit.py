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
def test_coalesce_no_limit(self):
    self.check([(10, 100, [(0, 10), (10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (60, 10), (70, 10), (80, 10), (90, 10)])], [(10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (60, 10), (70, 10), (80, 10), (90, 10), (100, 10)])