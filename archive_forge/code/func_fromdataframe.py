from __future__ import division, print_function, absolute_import
import inspect
from petl.util.base import Table
def fromdataframe(df, include_index=False):
    """
    Extract a table from a `pandas <http://pandas.pydata.org/>`_ DataFrame.
    E.g.::

        >>> import petl as etl
        >>> import pandas as pd
        >>> records = [('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
        >>> df = pd.DataFrame.from_records(records, columns=('foo', 'bar', 'baz'))
        >>> table = etl.fromdataframe(df)
        >>> table
        +-----------+-----+-----+
        | foo       | bar | baz |
        +===========+=====+=====+
        | 'apples'  |   1 | 2.5 |
        +-----------+-----+-----+
        | 'oranges' |   3 | 4.4 |
        +-----------+-----+-----+
        | 'pears'   |   7 | 0.1 |
        +-----------+-----+-----+

    """
    return DataFrameView(df, include_index=include_index)