import pytest
from panel.widgets.indicators import (
def test_dial_bounds():
    dial = Dial(bounds=(0, 20))
    with pytest.raises(ValueError):
        dial.value = 100