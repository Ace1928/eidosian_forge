import pytest
from panel.widgets.indicators import (
def test_gauge_bounds():
    dial = Gauge(bounds=(0, 20))
    with pytest.raises(ValueError):
        dial.value = 100