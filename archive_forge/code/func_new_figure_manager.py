import matplotlib
from matplotlib import colors
from matplotlib.backends import backend_agg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython
from IPython.core.pylabtools import select_figure_formats
from IPython.display import display
from .config import InlineBackend
def new_figure_manager(num, *args, FigureClass=Figure, **kwargs):
    """
    Return a new figure manager for a new figure instance.

    This function is part of the API expected by Matplotlib backends.
    """
    return new_figure_manager_given_figure(num, FigureClass(*args, **kwargs))