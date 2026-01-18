import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_integer_setitem():
    div = Div()
    gspec = GridSpec()
    gspec[0, 0] = div
    assert list(gspec.objects) == [(0, 0, 1, 1)]