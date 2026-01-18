from __future__ import division, print_function, absolute_import
import locale
from petl.compat import izip_longest, next, xrange, BytesIO
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def fromxls(filename, sheet=None, use_view=True, **kwargs):
    """
    Extract a table from a sheet in an Excel .xls file.
    
    Sheet is identified by its name or index number.
    
    N.B., the sheet name is case sensitive.

    """
    return XLSView(filename, sheet=sheet, use_view=use_view, **kwargs)