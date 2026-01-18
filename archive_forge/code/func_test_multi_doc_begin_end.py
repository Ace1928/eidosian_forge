import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load_all
def test_multi_doc_begin_end(self):
    from srsly.ruamel_yaml import dump_all, RoundTripDumper
    inp = '        ---\n        - a\n        ...\n        ---\n        - b\n        ...\n        '
    docs = list(round_trip_load_all(inp))
    assert docs == [['a'], ['b']]
    out = dump_all(docs, Dumper=RoundTripDumper, explicit_start=True, explicit_end=True)
    assert out == '---\n- a\n...\n---\n- b\n...\n'