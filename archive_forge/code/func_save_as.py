from a file, a socket or a WSGI environment. The parser can be used to replace
import re
import sys
from io import BytesIO
from tempfile import TemporaryFile
from urllib.parse import parse_qs
from wsgiref.headers import Headers
from collections.abc import MutableMapping as DictMixin
def save_as(self, path):
    with open(path, 'wb') as fp:
        pos = self.file.tell()
        try:
            self.file.seek(0)
            size = copy_file(self.file, fp)
        finally:
            self.file.seek(pos)
    return size