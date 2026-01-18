import unittest
from simplejson.compat import StringIO
import simplejson as json
def iter_dumps(obj, **kw):
    return ''.join(json.JSONEncoder(**kw).iterencode(obj))