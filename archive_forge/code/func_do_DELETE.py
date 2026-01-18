import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def do_DELETE(self):
    """Serve a DELETE request.

        We don't implement a true DELETE as DAV defines it
        because we *should* fail to delete a non empty dir.
        """
    path = self.translate_path(self.path)
    trace.mutter('do_DELETE rel: [{}], abs: [{}]'.format(self.path, path))
    try:
        real_path = os.path.realpath(path)
        if os.path.isdir(real_path):
            os.rmdir(path)
        else:
            os.remove(path)
    except OSError as e:
        if e.errno in (errno.ENOENT,):
            self.send_error(404, 'File not found')
        else:
            raise
    else:
        self.send_response(self.delete_success_code)
        self.end_headers()