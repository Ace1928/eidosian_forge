import os.path
import tempfile
from warnings import warn
import IPython
from IPython.utils.importstring import import_item
from IPython.utils.path import (
def get_ipython_cache_dir() -> str:
    """Get the cache directory it is created if it does not exist."""
    xdgdir = get_xdg_cache_dir()
    if xdgdir is None:
        return get_ipython_dir()
    ipdir = os.path.join(xdgdir, 'ipython')
    if not os.path.exists(ipdir) and _writable_dir(xdgdir):
        ensure_dir_exists(ipdir)
    elif not _writable_dir(xdgdir):
        return get_ipython_dir()
    return ipdir