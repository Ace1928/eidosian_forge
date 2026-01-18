import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def do_COPY(self):
    """Serve a COPY request."""
    url_to = self.headers.get('Destination')
    if url_to is None:
        self.send_error(400, 'Destination header missing')
        return
    scheme, netloc, rel_to, params, query, fragment = urllib.parse.urlparse(url_to)
    trace.mutter('urlparse: ({}) [{}]'.format(url_to, rel_to))
    trace.mutter('do_COPY rel_from: [{}], rel_to: [{}]'.format(self.path, rel_to))
    abs_from = self.translate_path(self.path)
    abs_to = self.translate_path(rel_to)
    try:
        shutil.copyfile(abs_from, abs_to)
    except OSError as e:
        if e.errno == errno.ENOENT:
            self.send_error(404, 'File not found')
        else:
            self.send_error(409, 'Conflict')
    else:
        self.send_response(201)
        self.end_headers()