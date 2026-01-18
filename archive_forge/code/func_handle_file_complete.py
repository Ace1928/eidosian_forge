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
def handle_file_complete(self, old_field_name, counters):
    """
        Handle all the signaling that takes place when a file is complete.
        """
    for i, handler in enumerate(self._upload_handlers):
        file_obj = handler.file_complete(counters[i])
        if file_obj:
            self._files.appendlist(force_str(old_field_name, self._encoding, errors='replace'), file_obj)
            break