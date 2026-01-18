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
def protocol_version(self):
    """Find out if 'hello' smart request works."""
    if self._protocol_version_error is not None:
        raise self._protocol_version_error
    if not self._done_hello:
        try:
            medium_request = self.get_request()
            client_protocol = protocol.SmartClientRequestProtocolOne(medium_request)
            client_protocol.query_version()
            self._done_hello = True
        except errors.SmartProtocolError as e:
            self._protocol_version_error = e
            raise
    return '2'