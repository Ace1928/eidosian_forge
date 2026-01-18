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
@hv_available
def test_holoviews_layouts(document, comm):
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['X', 'Y'])
    hv_pane = HoloViews(hmap, backend='bokeh')
    layout = hv_pane.layout
    model = layout.get_root(document, comm)
    for center in (True, False):
        for loc in HoloViews.param.widget_location.objects:
            hv_pane.param.update(center=center, widget_location=loc)
            if center:
                if loc.startswith('left'):
                    assert len(layout) == 4
                    widgets, hv_obj = (layout[0], layout[2])
                    wmodel, hv_model = (model.children[0], model.children[2])
                elif loc.startswith('right'):
                    assert len(layout) == 4
                    hv_obj, widgets = (layout[1], layout[3])
                    wmodel, hv_model = (model.children[3], model.children[1])
                elif loc.startswith('top'):
                    assert len(layout) == 3
                    col = layout[1]
                    cmodel = model.children[1]
                    assert isinstance(col, Column)
                    widgets, hv_obj = col
                    wmodel, hv_model = (cmodel.children[0], cmodel.children[1])
                elif loc.startswith('bottom'):
                    col = layout[1]
                    cmodel = model.children[1]
                    assert isinstance(col, Column)
                    hv_obj, widgets = col
                    wmodel, hv_model = (cmodel.children[1], cmodel.children[0])
            elif loc.startswith('left'):
                assert len(layout) == 2
                widgets, hv_obj = layout
                wmodel, hv_model = model.children
            elif loc.startswith('right'):
                assert len(layout) == 2
                hv_obj, widgets = layout
                hv_model, wmodel = model.children
            elif loc.startswith('top'):
                assert len(layout) == 1
                col = layout[0]
                cmodel = model.children[0]
                assert isinstance(col, Column)
                widgets, hv_obj = col
                wmodel, hv_model = cmodel.children
            elif loc.startswith('bottom'):
                assert len(layout) == 1
                col = layout[0]
                cmodel = model.children[0]
                assert isinstance(col, Column)
                hv_obj, widgets = col
                hv_model, wmodel = cmodel.children
            assert hv_pane is hv_obj
            assert isinstance(hv_model, figure)
            box = widgets
            boxmodel = wmodel
            assert hv_pane.widget_box is box
            assert isinstance(boxmodel, BkColumn)
            assert isinstance(boxmodel.children[0], BkColumn)
            assert isinstance(boxmodel.children[0].children[1], BkSlider)
            assert isinstance(boxmodel.children[1], BkSelect)