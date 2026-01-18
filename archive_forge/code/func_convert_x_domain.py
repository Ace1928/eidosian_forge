import math
import warnings
import matplotlib.dates
def convert_x_domain(mpl_plot_bounds, mpl_max_x_bounds):
    """Map x dimension of current plot to plotly's domain space.

    The bbox used to locate an axes object in mpl differs from the
    method used to locate axes in plotly. The mpl version locates each
    axes in the figure so that axes in a single-plot figure might have
    the bounds, [0.125, 0.125, 0.775, 0.775] (x0, y0, width, height),
    in mpl's figure coordinates. However, the axes all share one space in
    plotly such that the domain will always be [0, 0, 1, 1]
    (x0, y0, x1, y1). To convert between the two, the mpl figure bounds
    need to be mapped to a [0, 1] domain for x and y. The margins set
    upon opening a new figure will appropriately match the mpl margins.

    Optionally, setting margins=0 and simply copying the domains from
    mpl to plotly would place axes appropriately. However,
    this would throw off axis and title labeling.

    Positional arguments:
    mpl_plot_bounds -- the (x0, y0, width, height) params for current ax **
    mpl_max_x_bounds -- overall (x0, x1) bounds for all axes **

    ** these are all specified in mpl figure coordinates

    """
    mpl_x_dom = [mpl_plot_bounds[0], mpl_plot_bounds[0] + mpl_plot_bounds[2]]
    plotting_width = mpl_max_x_bounds[1] - mpl_max_x_bounds[0]
    x0 = (mpl_x_dom[0] - mpl_max_x_bounds[0]) / plotting_width
    x1 = (mpl_x_dom[1] - mpl_max_x_bounds[0]) / plotting_width
    return [x0, x1]