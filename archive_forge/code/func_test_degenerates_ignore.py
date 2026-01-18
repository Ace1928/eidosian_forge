import math
from unittest import TestCase
from simplejson.compat import long_type, text_type
import simplejson as json
from simplejson.decoder import NaN, PosInf, NegInf
def test_degenerates_ignore(self):
    for f in (PosInf, NegInf, NaN):
        self.assertEqual(json.loads(json.dumps(f, ignore_nan=True)), None)