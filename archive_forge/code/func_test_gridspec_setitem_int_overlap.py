import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_setitem_int_overlap():
    div = Div()
    gspec = GridSpec(mode='error')
    gspec[0, 0] = div
    with pytest.raises(IndexError):
        gspec[0, 0] = 'String'