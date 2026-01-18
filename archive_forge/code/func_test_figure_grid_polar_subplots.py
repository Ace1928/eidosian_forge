import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def test_figure_grid_polar_subplots(self):
    fig = figure_grid([[{'data': [{'type': 'scatterpolar', 'theta': [0, 90], 'r': [0.5, 1.0]}], 'layout': {'width': 450, 'height': 450}}], [{'data': [{'type': 'barpolar', 'theta': [90, 180], 'r': [1.0, 10.0]}], 'layout': {'width': 450, 'height': 450, 'polar': {'radialaxis': {'title': 'radial'}}}}]], row_spacing=100)
    go.Figure(fig)
    self.assertEqual(fig['data'][0]['type'], 'scatterpolar')
    self.assertEqual(fig['data'][0]['subplot'], 'polar')
    self.assertEqual(fig['layout']['polar']['domain'], {'y': [0, 0.45], 'x': [0, 1.0]})
    self.assertEqual(fig['data'][1]['type'], 'barpolar')
    self.assertEqual(fig['data'][1]['subplot'], 'polar2')
    self.assertEqual(fig['layout']['polar2']['domain'], {'y': [0.55, 1.0], 'x': [0, 1.0]})
    self.assertEqual(fig['layout']['width'], 450)
    self.assertEqual(fig['layout']['height'], 1000)
    self.assertEqual(fig['layout']['polar2']['radialaxis'], {'title': 'radial'})