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
def formataddr(pair, charset='utf-8'):
    """The inverse of parseaddr(), this takes a 2-tuple of the form
    (realname, email_address) and returns the string value suitable
    for an RFC 2822 From, To or Cc header.

    If the first element of pair is false, then the second element is
    returned unmodified.

    The optional charset is the character set that is used to encode
    realname in case realname is not ASCII safe.  Can be an instance of str or
    a Charset-like object which has a header_encode method.  Default is
    'utf-8'.
    """
    name, address = pair
    address.encode('ascii')
    if name:
        try:
            name.encode('ascii')
        except UnicodeEncodeError:
            if isinstance(charset, str):
                charset = Charset(charset)
            encoded_name = charset.header_encode(name)
            return '%s <%s>' % (encoded_name, address)
        else:
            quotes = ''
            if specialsre.search(name):
                quotes = '"'
            name = escapesre.sub('\\\\\\g<0>', name)
            return '%s%s%s <%s>' % (quotes, name, quotes, address)
    return address