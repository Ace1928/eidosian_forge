from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def parallel_categories(data_frame=None, dimensions=None, color=None, labels=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, title=None, template=None, width=None, height=None, dimensions_max_cardinality=50) -> go.Figure:
    """
    In a parallel categories (or parallel sets) plot, each row of
    `data_frame` is grouped with other rows that share the same values of
    `dimensions` and then plotted as a polyline mark through a set of
    parallel axes, one for each of the `dimensions`.
    """
    return make_figure(args=locals(), constructor=go.Parcats)