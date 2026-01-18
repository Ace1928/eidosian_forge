import os.path
import tempfile
from warnings import warn
import IPython
from IPython.utils.importstring import import_item
from IPython.utils.path import (
def get_ipython_package_dir() -> str:
    """Get the base directory where IPython itself is installed."""
    ipdir = os.path.dirname(IPython.__file__)
    assert isinstance(ipdir, str)
    return ipdir