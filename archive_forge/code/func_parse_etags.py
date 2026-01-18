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
def parse_etags(etag_str):
    """
    Parse a string of ETags given in an If-None-Match or If-Match header as
    defined by RFC 9110. Return a list of quoted ETags, or ['*'] if all ETags
    should be matched.
    """
    if etag_str.strip() == '*':
        return ['*']
    else:
        etag_matches = (ETAG_MATCH.match(etag.strip()) for etag in etag_str.split(','))
        return [match[1] for match in etag_matches if match]