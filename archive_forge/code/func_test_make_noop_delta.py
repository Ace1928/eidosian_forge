import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_make_noop_delta(self):
    ident_delta = self.make_delta(_text1, _text1)
    self.assertEqual(b'M\x90M', ident_delta)
    ident_delta = self.make_delta(_text2, _text2)
    self.assertEqual(b'N\x90N', ident_delta)
    ident_delta = self.make_delta(_text3, _text3)
    self.assertEqual(b'\x87\x01\x90\x87', ident_delta)