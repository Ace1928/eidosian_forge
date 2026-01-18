from .roundtrip import YAML
import pytest  # NOQA
def test_example_2_16():
    yaml = YAML()
    yaml.round_trip('\n    name: Mark McGwire\n    accomplishment: >\n      Mark set a major league\n      home run record in 1998.\n    stats: |\n      65 Home Runs\n      0.278 Batting Average\n    ')