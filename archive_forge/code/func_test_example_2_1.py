from .roundtrip import YAML
import pytest  # NOQA
def test_example_2_1():
    yaml = YAML()
    yaml.round_trip('\n    - Mark McGwire\n    - Sammy Sosa\n    - Ken Griffey\n    ')