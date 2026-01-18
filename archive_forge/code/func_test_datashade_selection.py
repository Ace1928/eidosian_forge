from unittest import skip, skipIf
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY
@ds_skip
def test_datashade_selection(self):
    points = Points(self.data)
    layout = points + dynspread(datashade(points))
    lnk_sel = link_selections.instance(unselected_color='#ff0000')
    linked = lnk_sel(layout)
    current_obj = linked[()]
    self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
    self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data)
    self.assertEqual(current_obj[1][()].RGB.I, dynspread(datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255))[()])
    self.assertEqual(current_obj[1][()].RGB.II, dynspread(datashade(points, cmap=lnk_sel.selected_cmap, alpha=255))[()])
    selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(selectionxy, SelectionXY)
    selectionxy.event(bounds=(0, 1, 5, 5))
    current_obj = linked[()]
    self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
    self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[1:])
    self.assertEqual(current_obj[1][()].RGB.I, dynspread(datashade(points, cmap=lnk_sel.unselected_cmap, alpha=255))[()])
    self.assertEqual(current_obj[1][()].RGB.II, dynspread(datashade(points.iloc[1:], cmap=lnk_sel.selected_cmap, alpha=255))[()])