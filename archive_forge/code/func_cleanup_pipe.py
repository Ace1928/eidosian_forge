import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def cleanup_pipe(self):
    """Read the remaining bytes of the last response if any."""
    if self._response is not None:
        try:
            pending = self._response.finish()
            if self._ranges_received_whole_file is None and self._response.status == 200 and pending and (pending > self._range_warning_thresold):
                self._ranges_received_whole_file = True
                trace.warning('Got a 200 response when asking for multiple ranges, does your server at %s:%s support range requests?', self.host, self.port)
        except OSError as e:
            if len(e.args) == 0 or e.args[0] not in (errno.ECONNRESET, errno.ECONNABORTED):
                raise
            self.close()
        self._response = None
    sock = self.sock
    self.sock = None
    self.close()
    self.sock = sock