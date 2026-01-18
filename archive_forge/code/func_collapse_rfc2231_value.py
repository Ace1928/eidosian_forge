import os
import re
import time
import random
import socket
import datetime
import urllib.parse
from email._parseaddr import quote
from email._parseaddr import AddressList as _AddressList
from email._parseaddr import mktime_tz
from email._parseaddr import parsedate, parsedate_tz, _parsedate_tz
from email.charset import Charset
def collapse_rfc2231_value(value, errors='replace', fallback_charset='us-ascii'):
    if not isinstance(value, tuple) or len(value) != 3:
        return unquote(value)
    charset, language, text = value
    if charset is None:
        charset = fallback_charset
    rawbytes = bytes(text, 'raw-unicode-escape')
    try:
        return str(rawbytes, charset, errors)
    except LookupError:
        return unquote(text)