import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_setitem_cell_override():
    div = Div()
    div2 = Div()
    gspec = GridSpec()
    gspec[0, 0] = div
    gspec[0, 0] = div2
    assert (0, 0, 1, 1) in gspec.objects
    assert gspec.objects[0, 0, 1, 1].object is div2