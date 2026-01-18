from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider', [EditableFloatSlider, EditableIntSlider])
@pytest.mark.parametrize('start,end,fixed_start,fixed_end,val_init,val_update,fail_init,fail_update', [(1, 5, 0, None, 2, 3, False, False), (1, 5, 0, None, 2, -1, False, True), (1, 5, 0, None, -1, -1, True, True), (1, 5, 0, None, 0, 100, False, False)])
def test_editable_slider_bounds(editableslider, start, end, fixed_start, fixed_end, val_init, val_update, fail_init, fail_update):
    try:
        slider = editableslider(start=start, end=end, fixed_start=fixed_start, fixed_end=fixed_end, value=val_init, name='Slider')
    except Exception:
        assert fail_init
    try:
        slider.value = val_update
    except Exception:
        assert fail_update