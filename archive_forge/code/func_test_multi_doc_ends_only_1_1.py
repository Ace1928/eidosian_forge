import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load_all
def test_multi_doc_ends_only_1_1(self):
    from srsly.ruamel_yaml import parser
    with pytest.raises(parser.ParserError):
        inp = '            - a\n            ...\n            - b\n            ...\n            '
        docs = list(round_trip_load_all(inp, version=(1, 1)))
        assert docs == [['a'], ['b']]