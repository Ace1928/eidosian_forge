from rpy2 import robjects
from rpy2.robjects.lib import ggplot2, grdevices
from IPython import get_ipython  # type: ignore
from IPython.core.display import Image  # type: ignore
def set_png_formatter():
    png_formatter = get_ipython().display_formatter.formatters['image/png']
    dpi = png_formatter.for_type(ggplot2.GGPlot, display_png)
    return dpi