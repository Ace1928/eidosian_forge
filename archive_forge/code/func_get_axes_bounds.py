import math
import warnings
import matplotlib.dates
def get_axes_bounds(fig):
    """Return the entire axes space for figure.

    An axes object in mpl is specified by its relation to the figure where
    (0,0) corresponds to the bottom-left part of the figure and (1,1)
    corresponds to the top-right. Margins exist in matplotlib because axes
    objects normally don't go to the edges of the figure.

    In plotly, the axes area (where all subplots go) is always specified with
    the domain [0,1] for both x and y. This function finds the smallest box,
    specified by two points, that all of the mpl axes objects fit into. This
    box is then used to map mpl axes domains to plotly axes domains.

    """
    x_min, x_max, y_min, y_max = ([], [], [], [])
    for axes_obj in fig.get_axes():
        bounds = axes_obj.get_position().bounds
        x_min.append(bounds[0])
        x_max.append(bounds[0] + bounds[2])
        y_min.append(bounds[1])
        y_max.append(bounds[1] + bounds[3])
    x_min, y_min, x_max, y_max = (min(x_min), min(y_min), max(x_max), max(y_max))
    return ((x_min, x_max), (y_min, y_max))