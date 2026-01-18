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
def do_crossfilter_points_histogram(self, selection_mode, cross_filter_mode, selected1, selected2, selected3, selected4, points_region1, points_region2, points_region3, points_region4, hist_region2, hist_region3, hist_region4, show_regions=True, dynamic=False):
    points = Points(self.data)
    hist = points.hist('x', adjoin=False, normed=False, num_bins=5)
    if dynamic:
        hist_orig = hist
        points = hv.util.Dynamic(points)
    else:
        hist_orig = hist
    lnk_sel = link_selections.instance(selection_mode=selection_mode, cross_filter_mode=cross_filter_mode, show_regions=show_regions, selected_color='#00ff00', unselected_color='#ff0000')
    linked = lnk_sel(points + hist)
    current_obj = linked[()]
    self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
    self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data)
    self.assertEqual(len(current_obj[0][()].Curve.I), 0)
    base_hist = current_obj[1][()].Histogram.I
    self.assertEqual(self.element_color(base_hist), lnk_sel.unselected_color)
    self.assertEqual(base_hist.data, hist_orig.data)
    selection_hist = current_obj[1][()].Histogram.II
    self.assertEqual(self.element_color(selection_hist), self.expected_selection_color(selection_hist, lnk_sel))
    self.assertEqual(selection_hist, base_hist)
    region_hist = current_obj[1][()].NdOverlay.I.last
    self.assertEqual(region_hist.data, [None, None])
    points_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(points_selectionxy, SelectionXY)
    points_selectionxy.event(bounds=(1, 1, 4, 4))
    current_obj = linked[()]
    self.check_base_points_like(current_obj[0][()].Points.I, lnk_sel)
    self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[selected1])
    region_bounds = current_obj[0][()].Rectangles.I
    self.assertEqual(region_bounds, Rectangles(points_region1))
    if show_regions:
        self.assertEqual(self.element_color(region_bounds), box_region_color)
    hist_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Histogram).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(hist_selectionxy, SelectionXY)
    hist_selectionxy.event(bounds=(0, 0, 2.5, 2))
    points_unsel, points_sel, points_region, points_region_poly = current_obj[0][()].values()
    self.check_overlay_points_like(points_sel, lnk_sel, self.data.iloc[selected2])
    self.assertEqual(points_region, Rectangles(points_region2))
    base_hist, region_hist, sel_hist = current_obj[1][()].values()
    self.assertEqual(self.element_color(base_hist), lnk_sel.unselected_color)
    self.assertEqual(base_hist.data, hist_orig.data)
    if show_regions:
        self.assertEqual(self.element_color(region_hist.last), hist_region_color)
    if not len(hist_region2) and lnk_sel.selection_mode != 'inverse':
        self.assertEqual(len(region_hist), 1)
    else:
        self.assertEqual(region_hist.last.data, [0, 2.5])
    self.assertEqual(self.element_color(sel_hist), self.expected_selection_color(sel_hist, lnk_sel))
    self.assertEqual(sel_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected2]).data)
    points_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Points).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(points_selectionxy, SelectionXY)
    points_selectionxy.event(bounds=(0, 0, 4, 2.5))
    self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[selected3])
    region_bounds = current_obj[0][()].Rectangles.I
    self.assertEqual(region_bounds, Rectangles(points_region3))
    selection_hist = current_obj[1][()].Histogram.II
    self.assertEqual(selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected3]).data)
    region_hist = current_obj[1][()].NdOverlay.I.last
    if not len(hist_region3) and lnk_sel.selection_mode != 'inverse':
        self.assertEqual(len(region_hist), 1)
    else:
        self.assertEqual(region_hist.data, [0, 2.5])
    hist_selectionxy = TestLinkSelections.get_value_with_key_type(lnk_sel._selection_expr_streams, hv.Histogram).input_streams[0].input_stream.input_streams[0]
    self.assertIsInstance(hist_selectionxy, SelectionXY)
    hist_selectionxy.event(bounds=(1.5, 0, 3.5, 2))
    self.check_overlay_points_like(current_obj[0][()].Points.II, lnk_sel, self.data.iloc[selected4])
    region_bounds = current_obj[0][()].Rectangles.I
    self.assertEqual(region_bounds, Rectangles(points_region4))
    region_hist = current_obj[1][()].NdOverlay.I.last
    if show_regions:
        self.assertEqual(self.element_color(region_hist), hist_region_color)
    if not len(hist_region4) and lnk_sel.selection_mode != 'inverse':
        self.assertEqual(len(region_hist), 1)
    elif lnk_sel.cross_filter_mode != 'overwrite' and (not (lnk_sel.cross_filter_mode == 'intersect' and lnk_sel.selection_mode == 'overwrite')):
        self.assertEqual(region_hist.data, [0, 3.5])
    else:
        self.assertEqual(region_hist.data, [1.5, 3.5])
    selection_hist = current_obj[1][()].Histogram.II
    self.assertEqual(self.element_color(selection_hist), self.expected_selection_color(selection_hist, lnk_sel))
    self.assertEqual(selection_hist.data, hist_orig.pipeline(hist_orig.dataset.iloc[selected4]).data)