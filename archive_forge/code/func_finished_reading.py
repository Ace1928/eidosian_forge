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
def finished_reading(self):
    """Inform the request that all desired data has been read.

        This will remove the request from the pipeline for its medium (if the
        medium supports pipelining) and any further calls to methods on the
        request will raise ReadingCompleted.
        """
    if self._state == 'writing':
        raise errors.WritingNotComplete(self)
    if self._state != 'reading':
        raise errors.ReadingCompleted(self)
    self._state = 'done'
    self._finished_reading()