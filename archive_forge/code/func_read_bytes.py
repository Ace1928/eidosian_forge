import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
def read_bytes(self, count):
    """Read bytes from this requests response.

        This method will block and wait for count bytes to be read. It may not
        be invoked until finished_writing() has been called - this is to ensure
        a message-based approach to requests, for compatibility with message
        based mediums like HTTP.
        """
    if self._state == 'writing':
        raise errors.WritingNotComplete(self)
    if self._state != 'reading':
        raise errors.ReadingCompleted(self)
    return self._read_bytes(count)