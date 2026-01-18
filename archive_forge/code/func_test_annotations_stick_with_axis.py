import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_annotations_stick_with_axis(self):
    fig = figure_grid([[{'data': [{'type': 'scatter', 'y': [1, 3, 2]}], 'layout': {'width': 400, 'height': 400, 'annotations': [{'text': 'One', 'xref': 'x', 'yref': 'y', 'x': 0, 'y': 0}, {'text': 'Two', 'xref': 'x', 'yref': 'y', 'x': 1, 'y': 0}]}}, {'data': [{'type': 'bar', 'y': [2, 3, 1]}], 'layout': {'width': 400, 'height': 400, 'annotations': [{'text': 'Three', 'xref': 'x', 'yref': 'y', 'x': 2, 'y': 0}, {'text': 'Four', 'xref': 'x', 'yref': 'y', 'x': 3, 'y': 0}]}}]])
    go.Figure(fig)
    annotations = fig['layout']['annotations']
    self.assertEqual(len(annotations), 4)
    self.assertEqual(annotations[0]['text'], 'One')
    self.assertEqual(annotations[0]['xref'], 'x')
    self.assertEqual(annotations[0]['yref'], 'y')
    self.assertEqual(annotations[1]['text'], 'Two')
    self.assertEqual(annotations[1]['xref'], 'x')
    self.assertEqual(annotations[1]['yref'], 'y')
    self.assertEqual(annotations[2]['text'], 'Three')
    self.assertEqual(annotations[2]['xref'], 'x2')
    self.assertEqual(annotations[2]['yref'], 'y2')
    self.assertEqual(annotations[3]['text'], 'Four')
    self.assertEqual(annotations[3]['xref'], 'x2')
    self.assertEqual(annotations[3]['yref'], 'y2')