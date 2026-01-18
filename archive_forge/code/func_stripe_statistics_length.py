from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
@property
def stripe_statistics_length(self):
    """The number of compressed bytes in the file stripe statistics"""
    return self.reader.stripe_statistics_length()