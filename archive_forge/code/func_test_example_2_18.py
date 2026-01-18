from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True, reason='non-literal/folding multiline scalars not supported')
def test_example_2_18():
    yaml = YAML()
    yaml.round_trip('\n    plain:\n      This unquoted scalar\n      spans many lines.\n\n    quoted: "So does this\n      quoted scalar.\n"\n    ')