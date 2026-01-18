import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_clone():
    div = Div()
    gspec = GridSpec()
    gspec[0, 0] = div
    clone = gspec.clone()
    assert gspec.objects == clone.objects
    assert gspec.param.values() == clone.param.values()