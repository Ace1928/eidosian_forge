import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def environ_decoder(key, default=_not_given, rfc_section=None, encattr=None):
    if rfc_section:
        doc = header_docstring(key, rfc_section)
    else:
        doc = 'Gets and sets the ``%s`` key in the environment.' % key
    if default is _not_given:

        def fget(req):
            return req.encget(key, encattr=encattr)

        def fset(req, val):
            return req.encset(key, val, encattr=encattr)
        fdel = None
    else:

        def fget(req):
            return req.encget(key, default, encattr=encattr)

        def fset(req, val):
            if val is None:
                if key in req.environ:
                    del req.environ[key]
            else:
                return req.encset(key, val, encattr=encattr)

        def fdel(req):
            del req.environ[key]
    return property(fget, fset, fdel, doc=doc)