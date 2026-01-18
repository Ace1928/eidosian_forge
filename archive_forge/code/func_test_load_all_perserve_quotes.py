from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_load_all_perserve_quotes(self):
    import srsly.ruamel_yaml
    s = dedent('        a: \'hello\'\n        ---\n        b: "goodbye"\n        ')
    data = []
    for x in srsly.ruamel_yaml.round_trip_load_all(s, preserve_quotes=True):
        data.append(x)
    out = srsly.ruamel_yaml.dump_all(data, Dumper=srsly.ruamel_yaml.RoundTripDumper)
    print(type(data[0]['a']), data[0]['a'])
    print(out)
    assert out == s