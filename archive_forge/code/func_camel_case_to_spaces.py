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
def camel_case_to_spaces(value):
    """
    Split CamelCase and convert to lowercase. Strip surrounding whitespace.
    """
    return re_camel_case.sub(' \\1', value).strip().lower()