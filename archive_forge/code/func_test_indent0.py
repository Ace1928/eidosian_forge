from unittest import TestCase
import textwrap
import simplejson as json
from simplejson.compat import StringIO
def test_indent0(self):
    h = {3: 1}

    def check(indent, expected):
        d1 = json.dumps(h, indent=indent)
        self.assertEqual(d1, expected)
        sio = StringIO()
        json.dump(h, sio, indent=indent)
        self.assertEqual(sio.getvalue(), expected)
    check(0, '{\n"3": 1\n}')
    check(None, '{"3": 1}')