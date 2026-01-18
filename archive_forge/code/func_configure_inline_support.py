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
def configure_inline_support(shell, backend):
    """Configure an IPython shell object for matplotlib use.

    Parameters
    ----------
    shell : InteractiveShell instance

    backend : matplotlib backend
    """
    cfg = InlineBackend.instance(parent=shell)
    cfg.shell = shell
    if cfg not in shell.configurables:
        shell.configurables.append(cfg)
    if backend == 'module://matplotlib_inline.backend_inline':
        shell.events.register('post_execute', flush_figures)
        shell._saved_rcParams = {}
        for k in cfg.rc:
            shell._saved_rcParams[k] = matplotlib.rcParams[k]
        matplotlib.rcParams.update(cfg.rc)
        new_backend_name = 'inline'
    else:
        try:
            shell.events.unregister('post_execute', flush_figures)
        except ValueError:
            pass
        if hasattr(shell, '_saved_rcParams'):
            matplotlib.rcParams.update(shell._saved_rcParams)
            del shell._saved_rcParams
        new_backend_name = 'other'
    cur_backend = getattr(configure_inline_support, 'current_backend', 'unset')
    if new_backend_name != cur_backend:
        select_figure_formats(shell, cfg.figure_formats, **cfg.print_figure_kwargs)
        configure_inline_support.current_backend = new_backend_name