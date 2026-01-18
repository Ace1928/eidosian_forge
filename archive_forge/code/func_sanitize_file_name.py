import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def sanitize_file_name(self, file_name):
    """
        Sanitize the filename of an upload.

        Remove all possible path separators, even though that might remove more
        than actually required by the target system. Filenames that could
        potentially cause problems (current/parent dir) are also discarded.

        It should be noted that this function could still return a "filepath"
        like "C:some_file.txt" which is handled later on by the storage layer.
        So while this function does sanitize filenames to some extent, the
        resulting filename should still be considered as untrusted user input.
        """
    file_name = html.unescape(file_name)
    file_name = file_name.rsplit('/')[-1]
    file_name = file_name.rsplit('\\')[-1]
    file_name = ''.join([char for char in file_name if char.isprintable()])
    if file_name in {'', '.', '..'}:
        return None
    return file_name