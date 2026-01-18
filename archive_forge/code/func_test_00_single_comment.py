import pytest  # NOQA
from .roundtrip import dedent, round_trip, round_trip_load
def test_00_single_comment(self):
    import srsly.ruamel_yaml
    s = dedent('        {}# Outside flow collection:\n        - ::vector\n        - ": - ()"\n        - Up, up, and away!\n        - -123\n        - http://example.com/foo#bar\n        - [::vector, ": - ()", "Down, down and away!", -456, http://example.com/foo#bar]\n        ')
    with pytest.raises(srsly.ruamel_yaml.parser.ParserError):
        round_trip(s.format('%YAML 1.1\n---\n'), preserve_quotes=True)
    round_trip(s.format(''), preserve_quotes=True)