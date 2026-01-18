import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_dump_unicode_utf8(self):
    import srsly.ruamel_yaml
    x = dedent(u'        ab:\n        - x  # comment\n        - y  # more comment\n        ')
    data = round_trip_load(x)
    dumper = srsly.ruamel_yaml.RoundTripDumper
    for utf in [True, False]:
        y = srsly.ruamel_yaml.dump(data, default_flow_style=False, Dumper=dumper, allow_unicode=utf)
        assert y == x