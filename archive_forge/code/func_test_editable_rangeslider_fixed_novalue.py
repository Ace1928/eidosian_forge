from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_editable_rangeslider_fixed_novalue():
    fixed_start, fixed_end, step = (5, 10, 0.01)
    slider = EditableRangeSlider(fixed_start=fixed_start, fixed_end=fixed_end)
    assert slider.value == (fixed_start, fixed_end)
    slider = EditableRangeSlider(fixed_start=fixed_start)
    assert slider.value == (fixed_start, fixed_start + 1)
    slider = EditableRangeSlider(fixed_start=fixed_start, step=step)
    assert slider.value == (fixed_start, fixed_start + step)
    slider = EditableRangeSlider(fixed_end=fixed_end)
    assert slider.value == (fixed_end - 1, fixed_end)
    slider = EditableRangeSlider(fixed_end=fixed_end, step=step)
    assert slider.value == (fixed_end - step, fixed_end)