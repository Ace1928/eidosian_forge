from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_13():
    yaml = YAML()
    yaml.round_trip('\n    # ASCII Art\n    --- |\n      \\//||\\/||\n      // ||  ||__\n    ')