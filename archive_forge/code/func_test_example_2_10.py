from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_10():
    yaml = YAML()
    yaml.explicit_start = True
    yaml.indent(sequence=4, offset=2)
    yaml.round_trip('\n    ---\n    hr:\n      - Mark McGwire\n      # Following node labeled SS\n      - &SS Sammy Sosa\n    rbi:\n      - *SS # Subsequent occurrence\n      - Ken Griffey\n    ')