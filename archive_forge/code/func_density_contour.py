from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def density_contour(data_frame=None, x=None, y=None, z=None, color=None, facet_row=None, facet_col=None, facet_col_wrap=0, facet_row_spacing=None, facet_col_spacing=None, hover_name=None, hover_data=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, orientation=None, color_discrete_sequence=None, color_discrete_map=None, marginal_x=None, marginal_y=None, trendline=None, trendline_options=None, trendline_color_override=None, trendline_scope='trace', log_x=False, log_y=False, range_x=None, range_y=None, histfunc=None, histnorm=None, nbinsx=None, nbinsy=None, text_auto=False, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a density contour plot, rows of `data_frame` are grouped together
    into contour marks to visualize the 2D distribution of an aggregate
    function `histfunc` (e.g. the count or sum) of the value `z`.
    """
    return make_figure(args=locals(), constructor=go.Histogram2dContour, trace_patch=dict(contours=dict(coloring='none'), histfunc=histfunc, histnorm=histnorm, nbinsx=nbinsx, nbinsy=nbinsy, xbingroup='x', ybingroup='y'))