import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_delta_with_delta_bytes(self):
    di = self._gc_module.DeltaIndex()
    source = _first_text
    di.add_source(_first_text, 0)
    self.assertEqual(len(_first_text), di._source_offset)
    delta = di.make_delta(_second_text)
    self.assertEqual(b'h\tsome more\x91\x019&previous text\nand has some extra text\n', delta)
    di.add_delta_source(delta, 0)
    source += delta
    self.assertEqual(len(_first_text) + len(delta), di._source_offset)
    second_delta = di.make_delta(_third_text)
    result = self._gc_module.apply_delta(source, second_delta)
    self.assertEqualDiff(_third_text, result)
    self.assertEqual(b'\x85\x01\x90\x14\x1chas some in common with the \x91S&\x03and\x91\x18,', second_delta)
    di.add_delta_source(second_delta, 0)
    source += second_delta
    third_delta = di.make_delta(_third_text)
    result = self._gc_module.apply_delta(source, third_delta)
    self.assertEqualDiff(_third_text, result)
    self.assertEqual(b'\x85\x01\x90\x14\x91~\x1c\x91S&\x03and\x91\x18,', third_delta)
    fourth_delta = di.make_delta(_fourth_text)
    self.assertEqual(_fourth_text, self._gc_module.apply_delta(source, fourth_delta))
    self.assertEqual(b'\x80\x01\x7f123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\x01\n', fourth_delta)
    di.add_delta_source(fourth_delta, 0)
    source += fourth_delta
    fifth_delta = di.make_delta(_fourth_text)
    self.assertEqual(_fourth_text, self._gc_module.apply_delta(source, fifth_delta))
    self.assertEqual(b'\x80\x01\x91\xa7\x7f\x01\n', fifth_delta)