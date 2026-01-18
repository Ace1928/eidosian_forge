from warnings import warn as _warn
import atexit
from . import version
from ._conv import register_converters as _register_converters, \
from .h5z import _register_lzf
from . import h5a, h5d, h5ds, h5f, h5fd, h5g, h5r, h5s, h5t, h5p, h5z, h5pl
from ._hl import filters
from ._hl.base import is_hdf5, HLObject, Empty
from ._hl.files import (
from ._hl.group import Group, SoftLink, ExternalLink, HardLink
from ._hl.dataset import Dataset
from ._hl.datatype import Datatype
from ._hl.attrs import AttributeManager
from ._hl.vds import VirtualSource, VirtualLayout
from ._selector import MultiBlockSlice
from .h5 import get_config
from .h5r import Reference, RegionReference
from .h5t import (special_dtype, check_dtype,
from .h5s import UNLIMITED
from .version import version as __version__
def enable_ipython_completer():
    """ Call this from an interactive IPython session to enable tab-completion
    of group and attribute names.
    """
    import sys
    if 'IPython' in sys.modules:
        ip_running = False
        try:
            from IPython.core.interactiveshell import InteractiveShell
            ip_running = InteractiveShell.initialized()
        except ImportError:
            from IPython import ipapi as _ipapi
            ip_running = _ipapi.get() is not None
        except Exception:
            pass
        if ip_running:
            from . import ipy_completer
            return ipy_completer.load_ipython_extension()
    raise RuntimeError('Completer must be enabled in active ipython session')