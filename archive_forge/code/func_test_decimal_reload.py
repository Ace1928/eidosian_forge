import decimal
from decimal import Decimal
from unittest import TestCase
from simplejson.compat import StringIO, reload_module
import simplejson as json
def test_decimal_reload(self):
    global Decimal
    Decimal = reload_module(decimal).Decimal
    import simplejson.encoder
    simplejson.encoder.Decimal = Decimal
    self.test_decimal_roundtrip()