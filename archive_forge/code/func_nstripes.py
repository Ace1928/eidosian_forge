from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def nstripes(self):
    """The number of stripes in the file"""
    return self.reader.nstripes()