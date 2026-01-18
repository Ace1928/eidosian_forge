from __future__ import (absolute_import, division,
from future.builtins import *
from future.backports import _markupbase
import re
import warnings
def replaceEntities(s):
    s = s.groups()[0]
    try:
        if s[0] == '#':
            s = s[1:]
            if s[0] in ['x', 'X']:
                c = int(s[1:].rstrip(';'), 16)
            else:
                c = int(s.rstrip(';'))
            return chr(c)
    except ValueError:
        return '&#' + s
    else:
        from future.backports.html.entities import html5
        if s in html5:
            return html5[s]
        elif s.endswith(';'):
            return '&' + s
        for x in range(2, len(s)):
            if s[:x] in html5:
                return html5[s[:x]] + s[x:]
        else:
            return '&' + s