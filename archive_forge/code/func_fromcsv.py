from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def fromcsv(source=None, encoding=None, errors='strict', header=None, **csvargs):
    """
    Extract a table from a delimited file. E.g.::

        >>> import petl as etl
        >>> import csv
        >>> # set up a CSV file to demonstrate with
        ... table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['c', 2]]
        >>> with open('example.csv', 'w') as f:
        ...     writer = csv.writer(f)
        ...     writer.writerows(table1)
        ...
        >>> # now demonstrate the use of fromcsv()
        ... table2 = etl.fromcsv('example.csv')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' | '1' |
        +-----+-----+
        | 'b' | '2' |
        +-----+-----+
        | 'c' | '2' |
        +-----+-----+

    The `source` argument is the path of the delimited file, all other keyword
    arguments are passed to :func:`csv.reader`. So, e.g., to override the
    delimiter from the default CSV dialect, provide the `delimiter` keyword
    argument.

    Note that all data values are strings, and any intended numeric values will
    need to be converted, see also :func:`petl.transform.conversions.convert`.

    """
    source = read_source_from_arg(source)
    csvargs.setdefault('dialect', 'excel')
    return fromcsv_impl(source=source, encoding=encoding, errors=errors, header=header, **csvargs)