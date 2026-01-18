from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True, reason='leading + on decimal dropped')
def test_example_2_19():
    yaml = YAML()
    yaml.round_trip('\n    canonical: 12345\n    decimal: +12345\n    octal: 0o14\n    hexadecimal: 0xC\n    ')