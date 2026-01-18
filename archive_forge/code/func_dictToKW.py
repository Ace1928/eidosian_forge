import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def dictToKW(d):
    out = []
    items = list(d.items())
    items.sort()
    for k, v in items:
        if not isinstance(k, str):
            raise NonFormattableDict("%r ain't a string" % k)
        if not r.match(k):
            raise NonFormattableDict("%r ain't an identifier" % k)
        out.append(f'\n\x00{k}={prettify(v)},')
    return ''.join(out)