import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
@pytest.fixture
def plotly_3d_plot():
    xx = np.linspace(-3.5, 3.5, 100)
    yy = np.linspace(-3.5, 3.5, 100)
    x, y = np.meshgrid(xx, yy)
    z = np.exp(-(x - 1) ** 2 - y ** 2) - (x ** 3 + y ** 4 - x / 5) * np.exp(-(x ** 2 + y ** 2))
    surface = go.Surface(z=z)
    title = 'Plotly 3D Plot'
    layout = go.Layout(title=title, autosize=False, width=500, height=500, margin=dict(t=50, b=50, r=50, l=50))
    fig = dict(data=[surface], layout=layout)
    plot_3d = Plotly(fig, width=500, height=500)
    return (plot_3d, title)