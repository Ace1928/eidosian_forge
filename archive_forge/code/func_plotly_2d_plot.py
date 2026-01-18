import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
@pytest.fixture
def plotly_2d_plot():
    trace = go.Scatter(x=[0, 1], y=[2, 3], uid='Test')
    plot_2d = Plotly({'data': [trace], 'layout': {'width': 350}})
    return plot_2d