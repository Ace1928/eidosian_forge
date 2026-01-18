import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def do_PROPFIND(self):
    """Serve a PROPFIND request."""
    depth = self.headers.get('Depth')
    if depth is None:
        depth = 'Infinity'
    if depth not in ('0', '1', 'Infinity'):
        self.send_error(400, 'Bad Depth')
        return
    self.read_body()
    try:
        response, st = self._generate_response(self.path)
    except OSError as e:
        if e.errno == errno.ENOENT:
            self.send_error(404)
            return
        else:
            raise
    if depth in ('1', 'Infinity') and stat.S_ISDIR(st.st_mode):
        dir_responses = self._generate_dir_responses(self.path, depth)
    else:
        dir_responses = []
    response = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="DAV:">\n{}{}\n</D:multistatus>'.format(response, ''.join(dir_responses)).encode('utf-8')
    self.send_response(207)
    self.send_header('Content-length', len(response))
    self.end_headers()
    self.wfile.write(response)