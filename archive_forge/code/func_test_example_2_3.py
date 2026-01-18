from .roundtrip import YAML
import pytest  # NOQA
def test_example_2_3():
    yaml = YAML()
    yaml.indent(sequence=4, offset=2)
    yaml.round_trip('\n    american:\n      - Boston Red Sox\n      - Detroit Tigers\n      - New York Yankees\n    national:\n      - New York Mets\n      - Chicago Cubs\n      - Atlanta Braves\n    ')