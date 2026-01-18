import decimal
from decimal import Decimal
from unittest import TestCase
from simplejson.compat import StringIO, reload_module
import simplejson as json
def test_decimal_decode(self):
    for s in self.NUMS:
        self.assertEqual(self.loads(s, parse_float=Decimal), Decimal(s))