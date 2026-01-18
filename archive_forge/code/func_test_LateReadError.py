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
def test_LateReadError(self):
    """The LateReadError helper should raise on read()."""
    a_file = transport.LateReadError('a path')
    try:
        a_file.read()
    except errors.ReadError as error:
        self.assertEqual('a path', error.path)
    self.assertRaises(errors.ReadError, a_file.read, 40)
    a_file.close()