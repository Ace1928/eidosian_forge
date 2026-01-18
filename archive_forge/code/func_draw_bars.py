import warnings
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib import mpltools
def draw_bars(self, bars):
    mpl_traces = []
    for container in self.bar_containers:
        mpl_traces.append([bar_props for bar_props in self.current_bars if bar_props['mplobj'] in container])
    for trace in mpl_traces:
        self.draw_bar(trace)