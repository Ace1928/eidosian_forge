import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def extract_name_value(line):
    """
    Return a list of (name, value) from a line of the form "name=value ...".

    :Exception:
        `NameValueError` for invalid input (missing name, missing data, bad
        quotes, etc.).
    """
    attlist = []
    while line:
        equals = line.find('=')
        if equals == -1:
            raise NameValueError('missing "="')
        attname = line[:equals].strip()
        if equals == 0 or not attname:
            raise NameValueError('missing attribute name before "="')
        line = line[equals + 1:].lstrip()
        if not line:
            raise NameValueError('missing value after "%s="' % attname)
        if line[0] in '\'"':
            endquote = line.find(line[0], 1)
            if endquote == -1:
                raise NameValueError('attribute "%s" missing end quote (%s)' % (attname, line[0]))
            if len(line) > endquote + 1 and line[endquote + 1].strip():
                raise NameValueError('attribute "%s" end quote (%s) not followed by whitespace' % (attname, line[0]))
            data = line[1:endquote]
            line = line[endquote + 1:].lstrip()
        else:
            space = line.find(' ')
            if space == -1:
                data = line
                line = ''
            else:
                data = line[:space]
                line = line[space + 1:].lstrip()
        attlist.append((attname.lower(), data))
    return attlist