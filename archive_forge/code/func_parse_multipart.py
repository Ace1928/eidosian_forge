from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
def parse_multipart(fp, pdict, encoding='utf-8', errors='replace', separator='&'):
    """Parse multipart input.

    Arguments:
    fp   : input file
    pdict: dictionary containing other parameters of content-type header
    encoding, errors: request encoding and error handler, passed to
        FieldStorage

    Returns a dictionary just like parse_qs(): keys are the field names, each
    value is a list of values for that field. For non-file fields, the value
    is a list of strings.
    """
    boundary = pdict['boundary'].decode('ascii')
    ctype = 'multipart/form-data; boundary={}'.format(boundary)
    headers = Message()
    headers.set_type(ctype)
    try:
        headers['Content-Length'] = pdict['CONTENT-LENGTH']
    except KeyError:
        pass
    fs = FieldStorage(fp, headers=headers, encoding=encoding, errors=errors, environ={'REQUEST_METHOD': 'POST'}, separator=separator)
    return {k: fs.getlist(k) for k in fs}