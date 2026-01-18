from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_4():
    yaml = YAML()
    yaml.mapping_value_align = True
    yaml.round_trip('\n    -\n      name: Mark McGwire\n      hr:   65\n      avg:  0.278\n    -\n      name: Sammy Sosa\n      hr:   63\n      avg:  0.288\n    ')