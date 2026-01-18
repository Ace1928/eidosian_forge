from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def teecsv(table, source=None, encoding=None, errors='strict', write_header=True, **csvargs):
    """
    Returns a table that writes rows to a CSV file as they are iterated over.

    """
    source = write_source_from_arg(source)
    csvargs.setdefault('dialect', 'excel')
    return teecsv_impl(table, source=source, encoding=encoding, errors=errors, write_header=write_header, **csvargs)