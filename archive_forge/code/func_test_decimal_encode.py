import decimal
from decimal import Decimal
from unittest import TestCase
from simplejson.compat import StringIO, reload_module
import simplejson as json
def test_decimal_encode(self):
    for d in map(Decimal, self.NUMS):
        self.assertEqual(self.dumps(d, use_decimal=True), str(d))