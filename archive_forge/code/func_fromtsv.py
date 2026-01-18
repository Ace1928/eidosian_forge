from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def fromtsv(source=None, encoding=None, errors='strict', header=None, **csvargs):
    """
    Convenience function, as :func:`petl.io.csv.fromcsv` but with different
    default dialect (tab delimited).

    """
    csvargs.setdefault('dialect', 'excel-tab')
    return fromcsv(source, encoding=encoding, errors=errors, header=header, **csvargs)