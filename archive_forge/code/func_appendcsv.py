from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def appendcsv(table, source=None, encoding=None, errors='strict', write_header=False, **csvargs):
    """
    Append data rows to an existing CSV file. As :func:`petl.io.csv.tocsv`
    but the file is opened in append mode and the table header is not written by
    default.

    Note that no attempt is made to check that the fields or row lengths are
    consistent with the existing data, the data rows from the table are simply
    appended to the file.

    """
    source = write_source_from_arg(source, mode='ab')
    csvargs.setdefault('dialect', 'excel')
    appendcsv_impl(table, source=source, encoding=encoding, errors=errors, write_header=write_header, **csvargs)