from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def scatter_matrix(data_frame=None, dimensions=None, color=None, symbol=None, size=None, hover_name=None, hover_data=None, custom_data=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, symbol_sequence=None, symbol_map=None, opacity=None, size_max=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a scatter plot matrix (or SPLOM), each row of `data_frame` is
    represented by a multiple symbol marks, one in each cell of a grid of
    2D scatter plots, which plot each pair of `dimensions` against each
    other.
    """
    return make_figure(args=locals(), constructor=go.Splom, layout_patch=dict(dragmode='select'))