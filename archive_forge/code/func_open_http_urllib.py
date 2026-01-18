import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def open_http_urllib(method, url, values):
    if not url:
        raise ValueError('cannot submit, no URL provided')
    try:
        from urllib import urlencode, urlopen
    except ImportError:
        from urllib.request import urlopen
        from urllib.parse import urlencode
    if method == 'GET':
        if '?' in url:
            url += '&'
        else:
            url += '?'
        url += urlencode(values)
        data = None
    else:
        data = urlencode(values)
        if not isinstance(data, bytes):
            data = data.encode('ASCII')
    return urlopen(url, data)