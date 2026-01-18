import os
import mimetypes
from typing import Generator
from libcloud.utils.py3 import b, next
def guess_file_mime_type(file_path):
    filename = os.path.basename(file_path)
    mimetype, encoding = mimetypes.guess_type(filename)
    return (mimetype, encoding)