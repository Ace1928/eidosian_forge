import copy
import pytest  # NOQA
from .roundtrip import dedent, round_trip_load, round_trip_dump
def test_copy_flow_style_nested_dict(self):
    x = dedent('        a: {foo: bar, baz: quux}\n        ')
    data = round_trip_load(x)
    assert data['a'].fa.flow_style() is True
    data_copy = copy.copy(data)
    assert data_copy['a'].fa.flow_style() is True
    data_copy['a'].fa.set_block_style()
    assert data['a'].fa.flow_style() == data_copy['a'].fa.flow_style()
    assert data['a'].fa._flow_style is False
    assert data_copy['a'].fa._flow_style is False
    y = round_trip_dump(data_copy)
    z = round_trip_dump(data)
    assert y == z
    assert y == dedent('        a:\n          foo: bar\n          baz: quux\n        ')