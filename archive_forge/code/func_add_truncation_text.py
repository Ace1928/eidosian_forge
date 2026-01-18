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
def add_truncation_text(text, truncate=None):
    if truncate is None:
        truncate = pgettext('String to return when truncating text', '%(truncated_text)s…')
    if '%(truncated_text)s' in truncate:
        return truncate % {'truncated_text': text}
    if text.endswith(truncate):
        return text
    return f'{text}{truncate}'