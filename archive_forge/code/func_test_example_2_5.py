from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_5():
    yaml = YAML()
    yaml.flow_sequence_element_align = True
    yaml.round_trip('\n    - [name        , hr, avg  ]\n    - [Mark McGwire, 65, 0.278]\n    - [Sammy Sosa  , 63, 0.288]\n    ')