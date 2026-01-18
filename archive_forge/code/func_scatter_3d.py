from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def scatter_3d(data_frame=None, x=None, y=None, z=None, color=None, symbol=None, size=None, text=None, hover_name=None, hover_data=None, custom_data=None, error_x=None, error_x_minus=None, error_y=None, error_y_minus=None, error_z=None, error_z_minus=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, size_max=None, color_discrete_sequence=None, color_discrete_map=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, symbol_sequence=None, symbol_map=None, opacity=None, log_x=False, log_y=False, log_z=False, range_x=None, range_y=None, range_z=None, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a 3D scatter plot, each row of `data_frame` is represented by a
    symbol mark in 3D space.
    """
    return make_figure(args=locals(), constructor=go.Scatter3d)