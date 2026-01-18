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
def flush_figures():
    """Send all figures that changed

    This is meant to be called automatically and will call show() if, during
    prior code execution, there had been any calls to draw_if_interactive.

    This function is meant to be used as a post_execute callback in IPython,
    so user-caused errors are handled with showtraceback() instead of being
    allowed to raise.  If this function is not called from within IPython,
    then these exceptions will raise.
    """
    if not show._draw_called:
        return
    try:
        if InlineBackend.instance().close_figures:
            try:
                return show(True)
            except Exception as e:
                ip = get_ipython()
                if ip is None:
                    raise e
                else:
                    ip.showtraceback()
                    return
        active = set([fm.canvas.figure for fm in Gcf.get_all_fig_managers()])
        for fig in [fig for fig in show._to_draw if fig in active]:
            try:
                display(fig, metadata=_fetch_figure_metadata(fig))
            except Exception as e:
                ip = get_ipython()
                if ip is None:
                    raise e
                else:
                    ip.showtraceback()
                    return
    finally:
        show._to_draw = []
        show._draw_called = False