import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_stretch_with_slice_setitem(document, comm):
    div1 = Div()
    div2 = Div()
    gspec = GridSpec(sizing_mode='stretch_both')
    gspec[0, 0:2] = div1
    gspec[1, 2] = div2
    model = gspec.get_root(document, comm=comm)
    assert model.children == [(div1, 0, 0, 1, 2), (div2, 1, 2, 1, 1)]
    assert div1.sizing_mode == 'stretch_both'
    assert div2.sizing_mode == 'stretch_both'