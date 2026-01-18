from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider', [EditableFloatSlider, EditableIntSlider, EditableRangeSlider])
def test_editable_fixed_nosoftbounds_fixed_end(editableslider):
    _, end = (8, 9)
    _, fixed_end = (5, 10)
    step = 2
    slider = editableslider(fixed_end=fixed_end)
    assert slider.start == fixed_end - 1
    assert slider.end == fixed_end
    slider = editableslider(fixed_end=fixed_end, step=step)
    assert slider.start == fixed_end - step
    assert slider.end == fixed_end
    slider = editableslider(fixed_end=fixed_end, end=end)
    assert slider.start == end - 1
    assert slider.end == end
    slider = editableslider(fixed_end=fixed_end, end=end, step=step)
    assert slider.start == end - step
    assert slider.end == end