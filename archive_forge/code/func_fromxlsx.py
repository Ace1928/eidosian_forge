from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import PY3, text_type
from petl.util.base import Table, data
from petl.io.sources import read_source_from_arg, write_source_from_arg
def fromxlsx(filename, sheet=None, range_string=None, min_row=None, min_col=None, max_row=None, max_col=None, read_only=False, **kwargs):
    """
    Extract a table from a sheet in an Excel .xlsx file.

    N.B., the sheet name is case sensitive.

    The `sheet` argument can be omitted, in which case the first sheet in
    the workbook is used by default.

    The `range_string` argument can be used to provide a range string
    specifying a range of cells to extract.

    The `min_row`, `min_col`, `max_row` and `max_col` arguments can be
    used to limit the range of cells to extract. They will be ignored
    if `range_string` is provided.

    The `read_only` argument determines how openpyxl returns the loaded 
    workbook. Default is `False` as it prevents some LibreOffice files
    from getting truncated at 65536 rows. `True` should be faster if the
    file use is read-only and the files are made with Microsoft Excel.

    Any other keyword arguments are passed through to
    :func:`openpyxl.load_workbook()`.

    """
    return XLSXView(filename, sheet=sheet, range_string=range_string, min_row=min_row, min_col=min_col, max_row=max_row, max_col=max_col, read_only=read_only, **kwargs)