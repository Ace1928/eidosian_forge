import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_setitem_slice_overlap():
    div = Div()
    gspec = GridSpec(mode='error')
    gspec[0, :] = div
    with pytest.raises(IndexError):
        gspec[0, 1] = div