import base64
import datetime
import re
import unicodedata
from binascii import Error as BinasciiError
from email.utils import formatdate
from urllib.parse import quote, unquote
from urllib.parse import urlencode as original_urlencode
from urllib.parse import urlparse
from django.utils.datastructures import MultiValueDict
from django.utils.regex_helper import _lazy_re_compile
def escape_leading_slashes(url):
    """
    If redirecting to an absolute path (two leading slashes), a slash must be
    escaped to prevent browsers from handling the path as schemaless and
    redirecting to another host.
    """
    if url.startswith('//'):
        url = '/%2F{}'.format(url.removeprefix('//'))
    return url