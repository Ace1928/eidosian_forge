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
def test__filter(self):
    t = self.make_pf_transport()
    self.assertEqual('foo', t._filter('foo'))
    self.assertEqual('foo/bar', t._filter('foo/bar'))
    self.assertEqual('', t._filter('..'))
    self.assertEqual('', t._filter('/'))
    t = t.clone('subdir1/subdir2')
    self.assertEqual('subdir1/subdir2/foo', t._filter('foo'))
    self.assertEqual('subdir1/subdir2/foo/bar', t._filter('foo/bar'))
    self.assertEqual('subdir1', t._filter('..'))
    self.assertEqual('', t._filter('/'))