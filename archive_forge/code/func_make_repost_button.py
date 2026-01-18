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
def make_repost_button(environ):
    url = request.construct_url(environ)
    if environ['REQUEST_METHOD'] == 'GET':
        return '<button onclick="window.location.href=%r">Re-GET Page</button><br>' % url
    else:
        return None
    '\n    fields = []\n    for name, value in wsgilib.parse_formvars(\n        environ, include_get_vars=False).items():\n        if hasattr(value, \'filename\'):\n            # @@: Arg, we\'ll just submit the body, and leave out\n            # the filename :(\n            value = value.value\n        fields.append(\n            \'<input type="hidden" name="%s" value="%s">\'\n            % (html_quote(name), html_quote(value)))\n    return \'\'\'\n<form action="%s" method="POST">\n%s\n<input type="submit" value="Re-POST Page">\n</form>\'\'\' % (url, \'\n\'.join(fields))\n'