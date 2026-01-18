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
def base36_to_int(s):
    """
    Convert a base 36 string to an int. Raise ValueError if the input won't fit
    into an int.
    """
    if len(s) > 13:
        raise ValueError('Base36 input too large')
    return int(s, 36)