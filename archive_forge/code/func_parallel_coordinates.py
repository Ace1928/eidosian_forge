from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def parallel_coordinates(data_frame=None, dimensions=None, color=None, labels=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a parallel coordinates plot, each row of `data_frame` is represented
    by a polyline mark which traverses a set of parallel axes, one for each
    of the `dimensions`.
    """
    return make_figure(args=locals(), constructor=go.Parcoords)