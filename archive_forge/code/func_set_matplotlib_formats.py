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
def set_matplotlib_formats(*formats, **kwargs):
    """Select figure formats for the inline backend. Optionally pass quality for JPEG.

    For example, this enables PNG and JPEG output with a JPEG quality of 90%::

        In [1]: set_matplotlib_formats('png', 'jpeg', quality=90)

    To set this in your config files use the following::

        c.InlineBackend.figure_formats = {'png', 'jpeg'}
        c.InlineBackend.print_figure_kwargs.update({'quality' : 90})

    Parameters
    ----------
    *formats : strs
        One or more figure formats to enable: 'png', 'retina', 'jpeg', 'svg', 'pdf'.
    **kwargs
        Keyword args will be relayed to ``figure.canvas.print_figure``.
    """
    cfg = InlineBackend.instance()
    kw = {}
    kw.update(cfg.print_figure_kwargs)
    kw.update(**kwargs)
    shell = InteractiveShell.instance()
    select_figure_formats(shell, formats, **kw)