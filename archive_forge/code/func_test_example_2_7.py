from .roundtrip import YAML
import pytest  # NOQA
@pytest.mark.xfail(strict=True)
def test_example_2_7():
    yaml = YAML()
    yaml.round_trip_all('\n    # Ranking of 1998 home runs\n    ---\n    - Mark McGwire\n    - Sammy Sosa\n    - Ken Griffey\n\n    # Team ranking\n    ---\n    - Chicago Cubs\n    - St Louis Cardinals\n    ')