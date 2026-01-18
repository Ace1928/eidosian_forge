import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_make_delta_is_typesafe(self):
    self.make_delta(b'a string', b'another string')

    def _check_make_delta(string1, string2):
        self.assertRaises(TypeError, self.make_delta, string1, string2)
    _check_make_delta(b'a string', object())
    _check_make_delta(b'a string', 'not a string')
    _check_make_delta(object(), b'a string')
    _check_make_delta('not a string', b'a string')