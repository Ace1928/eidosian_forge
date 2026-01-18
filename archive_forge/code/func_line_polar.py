from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def line_polar(data_frame=None, r=None, theta=None, color=None, line_dash=None, hover_name=None, hover_data=None, custom_data=None, line_group=None, text=None, symbol=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, color_discrete_sequence=None, color_discrete_map=None, line_dash_sequence=None, line_dash_map=None, symbol_sequence=None, symbol_map=None, markers=False, direction='clockwise', start_angle=90, line_close=False, line_shape=None, render_mode='auto', range_r=None, range_theta=None, log_r=False, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a polar line plot, each row of `data_frame` is represented as vertex
    of a polyline mark in polar coordinates.
    """
    return make_figure(args=locals(), constructor=go.Scatterpolar)