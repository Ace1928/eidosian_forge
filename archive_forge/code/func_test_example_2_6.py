from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_6():
    yaml = YAML()
    yaml.flow_mapping_one_element_per_line = True
    yaml.round_trip('\n    Mark McGwire: {hr: 65, avg: 0.278}\n    Sammy Sosa: {\n        hr: 63,\n        avg: 0.288\n      }\n    ')