from datetime import datetime
import pytest
from cartopy.feature.nightshade import Nightshade, _julian_day, _solar_position
def test_julian_day():
    dt = datetime(1996, 10, 26, 14, 20)
    jd = _julian_day(dt)
    assert pytest.approx(jd) == 2450383.09722222