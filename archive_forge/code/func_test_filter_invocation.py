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
def test_filter_invocation(self):
    filter_log = []

    def filter(path):
        filter_log.append(path)
        return path
    t = self.make_pf_transport(filter)
    t.has('abc')
    self.assertEqual(['abc'], filter_log)
    del filter_log[:]
    t.clone('abc').has('xyz')
    self.assertEqual(['abc/xyz'], filter_log)
    del filter_log[:]
    t.has('/abc')
    self.assertEqual(['abc'], filter_log)