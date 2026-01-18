import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def do_PUT(self):
    """Serve a PUT request."""
    path = self.translate_path(self.path)
    trace.mutter('do_PUT rel: [{}], abs: [{}]'.format(self.path, path))
    do_append = False
    range_header = self.headers.get('Content-Range')
    if range_header is not None:
        match = self._RANGE_HEADER_RE.match(range_header)
        if match is None:
            self.send_error(501, 'Not Implemented')
            return
        begin = int(match.group('begin'))
        do_append = True
    if self.headers.get('Expect') == '100-continue':
        self.send_response(100, 'Continue')
        self.end_headers()
    try:
        trace.mutter('do_PUT will try to open: [%s]' % path)
        if do_append:
            f = open(path, 'ab')
            f.seek(begin)
        else:
            f = open(path, 'wb')
    except OSError as e:
        trace.mutter('do_PUT got: [%r] while opening/seeking on [%s]' % (e, self.path))
        self.send_error(409, 'Conflict')
        return
    try:
        data = self.read_body()
        f.write(data)
    except OSError:
        self.send_error(409, 'Conflict')
        f.close()
        return
    f.close()
    trace.mutter('do_PUT done: [%s]' % self.path)
    self.send_response(201)
    self.end_headers()