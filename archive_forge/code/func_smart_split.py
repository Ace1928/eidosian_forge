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
def smart_split(text):
    """
    Generator that splits a string by spaces, leaving quoted phrases together.
    Supports both single and double quotes, and supports escaping quotes with
    backslashes. In the output, strings will keep their initial and trailing
    quote marks and escaped quotes will remain escaped (the results can then
    be further processed with unescape_string_literal()).

    >>> list(smart_split(r'This is "a person\\'s" test.'))
    ['This', 'is', '"a person\\\\\\'s"', 'test.']
    >>> list(smart_split(r"Another 'person\\'s' test."))
    ['Another', "'person\\\\'s'", 'test.']
    >>> list(smart_split(r'A "\\"funky\\" style" test.'))
    ['A', '"\\\\"funky\\\\" style"', 'test.']
    """
    for bit in smart_split_re.finditer(str(text)):
        yield bit[0]