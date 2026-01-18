import warnings
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib import mpltools
def strip_style(self):
    self.msg += 'Stripping mpl style is no longer supported\n'