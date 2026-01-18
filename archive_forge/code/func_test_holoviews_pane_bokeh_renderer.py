import datetime as dt
import warnings
import numpy as np
import pytest
from bokeh.models import (
from bokeh.plotting import figure
import panel as pn
from panel.depends import bind
from panel.layout import (
from panel.pane import HoloViews, PaneBase, panel
from panel.tests.util import hv_available, mpl_available
from panel.theme import Native
from panel.util.warnings import PanelDeprecationWarning
from panel.widgets import (
@pytest.mark.usefixtures('hv_bokeh')
@hv_available
def test_holoviews_pane_bokeh_renderer(document, comm):
    curve = hv.Curve([1, 2, 3])
    pane = pn.panel(curve)
    row = pane.get_root(document, comm=comm)
    assert isinstance(row, BkRow)
    assert len(row.children) == 1
    model = row.children[0]
    assert isinstance(model, figure)
    assert pane._models[row.ref['id']][0] is model
    renderers = [r for r in model.renderers if isinstance(r, GlyphRenderer)]
    assert len(renderers) == 1
    assert isinstance(renderers[0].glyph, Line)
    scatter = hv.Scatter([1, 2, 3])
    pane.object = scatter
    model = row.children[0]
    assert isinstance(model, figure)
    renderers = [r for r in model.renderers if isinstance(r, GlyphRenderer)]
    assert len(renderers) == 1
    assert isinstance(renderers[0].glyph, Scatter)
    assert pane._models[row.ref['id']][0] is model
    pane._cleanup(row)
    assert pane._models == {}