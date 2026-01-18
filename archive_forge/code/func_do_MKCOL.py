import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def do_MKCOL(self):
    """
        Serve a MKCOL request.

        MKCOL is an mkdir in DAV terminology for our part.
        """
    path = self.translate_path(self.path)
    trace.mutter('do_MKCOL rel: [{}], abs: [{}]'.format(self.path, path))
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno in (errno.ENOENT,):
            self.send_error(409, 'Conflict')
        elif e.errno in (errno.EEXIST, errno.ENOTDIR):
            self.send_error(405, 'Not allowed')
        else:
            raise
    else:
        self.send_response(201)
        self.end_headers()