import sys
import traceback
from contextlib import contextmanager
from functools import wraps
import IPython
from IPython import get_ipython
from IPython.display import HTML
import holoviews as hv
from ..core import (
from ..core.io import FileArchive
from ..core.options import AbbreviatedException, SkipRendering, Store, StoreOptions
from ..core.traversal import unique_dimkeys
from ..core.util import mimebundle_to_html
from ..plotting import Plot
from ..plotting.renderer import MIME_TYPES
from ..util.settings import OutputSettings
from .magics import OptsMagic, OutputMagic
def single_frame_plot(obj):
    """
    Returns plot, renderer and format for single frame export.
    """
    obj = Layout(obj) if isinstance(obj, AdjointLayout) else obj
    backend = Store.current_backend
    renderer = Store.renderers[backend]
    plot_cls = renderer.plotting_class(obj)
    plot = plot_cls(obj, **renderer.plot_options(obj, renderer.size))
    fmt = renderer.param.objects('existing')['fig'].objects[0] if renderer.fig == 'auto' else renderer.fig
    return (plot, renderer, fmt)