import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def serialize_etag_response(value):
    strong = True
    if isinstance(value, tuple):
        value, strong = value
    elif _rx_etag.match(value):
        return value
    r = '"%s"' % value.replace('"', '\\"')
    if not strong:
        r = 'W/' + r
    return r