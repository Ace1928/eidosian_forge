import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
def preserve_whitespace(v, quote=True):
    """
    Quote a value for HTML, preserving whitespace (translating
    newlines to ``<br>`` and multiple spaces to use ``&nbsp;``).

    If ``quote`` is true, then the value will be HTML quoted first.
    """
    if quote:
        v = html_quote(v)
    v = v.replace('\n', '<br>\n')
    v = re.sub('()(  +)', _repl_nbsp, v)
    v = re.sub('(\\n)( +)', _repl_nbsp, v)
    v = re.sub('^()( +)', _repl_nbsp, v)
    return '<code>%s</code>' % v