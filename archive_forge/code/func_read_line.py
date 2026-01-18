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
def read_line(self):
    line = self._read_line()
    if not line.endswith(b'\n'):
        raise errors.ConnectionReset('Unexpected end of message. Please check connectivity and permissions, and report a bug if problems persist.')
    return line