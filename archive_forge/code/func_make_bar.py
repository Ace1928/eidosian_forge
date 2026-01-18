import math
import warnings
import matplotlib.dates
def make_bar(**props):
    """Make an intermediate bar dictionary.

    This creates a bar dictionary which aids in the comparison of new bars to
    old bars from other bar chart (patch) collections. This is not the
    dictionary that needs to get passed to plotly as a data dictionary. That
    happens in PlotlyRenderer in that class's draw_bar method. In other
    words, this dictionary describes a SINGLE bar, whereas, plotly will
    require a set of bars to be passed in a data dictionary.

    """
    return {'bar': props['mplobj'], 'x0': get_rect_xmin(props['data']), 'y0': get_rect_ymin(props['data']), 'x1': get_rect_xmax(props['data']), 'y1': get_rect_ymax(props['data']), 'alpha': props['style']['alpha'], 'edgecolor': props['style']['edgecolor'], 'facecolor': props['style']['facecolor'], 'edgewidth': props['style']['edgewidth'], 'dasharray': props['style']['dasharray'], 'zorder': props['style']['zorder']}