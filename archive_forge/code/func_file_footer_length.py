from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def file_footer_length(self):
    """The number of compressed bytes in the file footer"""
    return self.reader.file_footer_length()