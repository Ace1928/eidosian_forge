import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_fixed_with_int_setitem(document, comm):
    div1 = Div()
    div2 = Div()
    gspec = GridSpec(width=800, height=500)
    gspec[0, 0] = div1
    gspec[1, 1] = div2
    model = gspec.get_root(document, comm=comm)
    assert model.children == [(div1, 0, 0, 1, 1), (div2, 1, 1, 1, 1)]
    assert div1.width == 400
    assert div1.height == 250
    assert div2.width == 400
    assert div2.height == 250