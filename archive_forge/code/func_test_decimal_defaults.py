import decimal
from decimal import Decimal
from unittest import TestCase
from simplejson.compat import StringIO, reload_module
import simplejson as json
def test_decimal_defaults(self):
    d = Decimal('1.1')
    self.assertRaises(TypeError, json.dumps, d, use_decimal=False)
    self.assertEqual('1.1', json.dumps(d))
    self.assertEqual('1.1', json.dumps(d, use_decimal=True))
    self.assertRaises(TypeError, json.dump, d, StringIO(), use_decimal=False)
    sio = StringIO()
    json.dump(d, sio)
    self.assertEqual('1.1', sio.getvalue())
    sio = StringIO()
    json.dump(d, sio, use_decimal=True)
    self.assertEqual('1.1', sio.getvalue())