from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_colorbar_opts_title(self):
    df = pd.DataFrame(np.random.random((10, 4)), columns=list('XYZT'))
    scatter = Scatter3D(data=df).opts(color='T', colorbar=True, colorbar_opts={'title': 'some-title'})
    state = self._get_plot_state(scatter)
    assert 'colorbar' in state['data'][0]['marker']
    assert state['data'][0]['marker']['colorbar']['title']['text'] == 'some-title'