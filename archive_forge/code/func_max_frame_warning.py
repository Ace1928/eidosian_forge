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
def max_frame_warning(max_frames):
    sys.stderr.write(f'Animation longer than the max_frames limit {max_frames};\nskipping rendering to avoid unexpected lengthy computations.\nIf desired, the limit can be increased using:\nhv.output(max_frames=<insert number>)')