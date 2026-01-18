import unittest
from simplejson.compat import StringIO
import simplejson as json
def sio_dump(obj, **kw):
    sio = StringIO()
    json.dumps(obj, **kw)
    return sio.getvalue()