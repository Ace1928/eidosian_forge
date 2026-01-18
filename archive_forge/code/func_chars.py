import gzip
import re
import secrets
import unicodedata
from gzip import GzipFile
from gzip import compress as gzip_compress
from io import BytesIO
from django.core.exceptions import SuspiciousFileOperation
from django.utils.functional import SimpleLazyObject, keep_lazy_text, lazy
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy, pgettext
def chars(self, num, truncate=None, html=False):
    """
        Return the text truncated to be no longer than the specified number
        of characters.

        `truncate` specifies what should be used to notify that the string has
        been truncated, defaulting to a translatable string of an ellipsis.
        """
    self._setup()
    length = int(num)
    text = unicodedata.normalize('NFC', self._wrapped)
    truncate_len = length
    for char in add_truncation_text('', truncate):
        if not unicodedata.combining(char):
            truncate_len -= 1
            if truncate_len == 0:
                break
    if html:
        return self._truncate_html(length, truncate, text, truncate_len, False)
    return self._text_chars(length, truncate, text, truncate_len)