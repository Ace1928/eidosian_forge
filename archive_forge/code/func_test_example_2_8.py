from .roundtrip import YAML
import pytest  # NOQA
def test_example_2_8():
    yaml = YAML()
    yaml.explicit_start = True
    yaml.explicit_end = True
    yaml.round_trip_all('\n    ---\n    time: 20:03:20\n    player: Sammy Sosa\n    action: strike (miss)\n    ...\n    ---\n    time: 20:03:47\n    player: Sammy Sosa\n    action: grand slam\n    ...\n    ')