from ._core import make_figure
from ._doc import make_docstring
import plotly.graph_objs as go
def density_heatmap(data_frame=None, x=None, y=None, z=None, facet_row=None, facet_col=None, facet_col_wrap=0, facet_row_spacing=None, facet_col_spacing=None, hover_name=None, hover_data=None, animation_frame=None, animation_group=None, category_orders=None, labels=None, orientation=None, color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, marginal_x=None, marginal_y=None, opacity=None, log_x=False, log_y=False, range_x=None, range_y=None, histfunc=None, histnorm=None, nbinsx=None, nbinsy=None, text_auto=False, title=None, template=None, width=None, height=None) -> go.Figure:
    """
    In a density heatmap, rows of `data_frame` are grouped together into
    colored rectangular tiles to visualize the 2D distribution of an
    aggregate function `histfunc` (e.g. the count or sum) of the value `z`.
    """
    return make_figure(args=locals(), constructor=go.Histogram2d, trace_patch=dict(histfunc=histfunc, histnorm=histnorm, nbinsx=nbinsx, nbinsy=nbinsy, xbingroup='x', ybingroup='y'))