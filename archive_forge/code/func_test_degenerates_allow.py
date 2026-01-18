import math
from unittest import TestCase
from simplejson.compat import long_type, text_type
import simplejson as json
from simplejson.decoder import NaN, PosInf, NegInf
def test_degenerates_allow(self):
    for inf in (PosInf, NegInf):
        self.assertEqual(json.loads(json.dumps(inf, allow_nan=True), allow_nan=True), inf)
    nan = json.loads(json.dumps(NaN, allow_nan=True), allow_nan=True)
    self.assertTrue(0 + nan != nan)