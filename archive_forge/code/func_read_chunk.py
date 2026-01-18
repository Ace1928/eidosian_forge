import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def read_chunk(self):
    """Read a chunk of data.

        A chunk consists of:
        - a line containing the length of the data in hexa,
        - the data.
        - a empty line.

        An empty chunk specifies a length of zero
        """
    length = int(self._readline(), 16)
    data = None
    if length != 0:
        data = self._read(length)
        self._readline()
    return (length, data)