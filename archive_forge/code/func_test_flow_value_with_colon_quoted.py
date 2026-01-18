import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_flow_value_with_colon_quoted(self):
    inp = "        {a: 'bcd:efg'}\n        "
    round_trip(inp, preserve_quotes=True)