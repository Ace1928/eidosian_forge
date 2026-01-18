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
def finished_writing(self):
    """Finish the writing phase of this request.

        This will flush all pending data for this request along the medium.
        After calling finished_writing, you may not call accept_bytes anymore.
        """
    if self._state != 'writing':
        raise errors.WritingCompleted(self)
    self._state = 'reading'
    self._finished_writing()