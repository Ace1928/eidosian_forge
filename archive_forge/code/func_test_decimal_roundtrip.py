import decimal
from decimal import Decimal
from unittest import TestCase
from simplejson.compat import StringIO, reload_module
import simplejson as json
def test_decimal_roundtrip(self):
    for d in map(Decimal, self.NUMS):
        for v in [d, [d], {'': d}]:
            self.assertEqual(self.loads(self.dumps(v, use_decimal=True), parse_float=Decimal), v)