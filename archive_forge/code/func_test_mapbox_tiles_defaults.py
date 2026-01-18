import numpy as np
import pytest
from holoviews.element import RGB, Bounds, Points, Tiles
from holoviews.element.tiles import _ATTRIBUTIONS, StamenTerrain
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_mapbox_tiles_defaults(self):
    tiles = Tiles('').redim.range(x=self.x_range, y=self.y_range)
    fig_dict = plotly_renderer.get_plot_state(tiles)
    self.assertEqual(len(fig_dict['data']), 1)
    dummy_trace = fig_dict['data'][0]
    self.assertEqual(dummy_trace['type'], 'scattermapbox')
    self.assertEqual(dummy_trace['lon'], [])
    self.assertEqual(dummy_trace['lat'], [])
    self.assertEqual(dummy_trace['showlegend'], False)
    subplot = fig_dict['layout']['mapbox']
    self.assertEqual(subplot['style'], 'white-bg')
    self.assertEqual(subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center})
    self.assertNotIn('xaxis', fig_dict['layout'])
    self.assertNotIn('yaxis', fig_dict['layout'])
    layers = fig_dict['layout']['mapbox'].get('layers', [])
    self.assertEqual(len(layers), 0)