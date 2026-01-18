from holoviews import Overlay
from holoviews.annotators import PathAnnotator, PointAnnotator, annotate
from holoviews.element import Path, Points, Table
from holoviews.element.tiles import EsriStreet, Tiles
from holoviews.tests.plotting.bokeh.test_plot import TestBokehPlot
def test_annotate_overlay(self):
    layout = annotate(EsriStreet() * Points([]), annotations=['Label'])
    overlay = layout.DynamicMap.I[()]
    tables = layout.Annotator.PointAnnotator[()]
    self.assertIsInstance(overlay, Overlay)
    self.assertEqual(len(overlay), 2)
    self.assertIsInstance(overlay.get(0), Tiles)
    self.assertEqual(overlay.get(1), Points([], vdims='Label'))
    self.assertIsInstance(tables, Overlay)
    self.assertEqual(len(tables), 1)