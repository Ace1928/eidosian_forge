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
@display_hook
def layout_display(layout, max_frames):
    if isinstance(layout, AdjointLayout):
        layout = Layout(layout).opts(layout.opts.get('plot'))
    if not isinstance(layout, (Layout, NdLayout)):
        return None
    nframes = len(unique_dimkeys(layout)[1])
    if nframes > max_frames:
        max_frame_warning(max_frames)
        return None
    return render(layout)