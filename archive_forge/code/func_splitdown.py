from __future__ import absolute_import, print_function, division
import re
import operator
from petl.compat import next, text_type
from petl.errors import ArgumentError
from petl.util.base import Table, asindices
from petl.transform.basics import TransformError
from petl.transform.conversions import convert
def splitdown(table, field, pattern, maxsplit=0, flags=0):
    """
    Split a field into multiple rows using a regular expression. E.g.:

        >>> import petl as etl
        >>> table1 = [['name', 'roles'],
        ...           ['Jane Doe', 'president,engineer,tailor,lawyer'],
        ...           ['John Doe', 'rocket scientist,optometrist,chef,knight,sailor']]
        >>> table2 = etl.splitdown(table1, 'roles', ',')
        >>> table2.lookall()
        +------------+--------------------+
        | name       | roles              |
        +============+====================+
        | 'Jane Doe' | 'president'        |
        +------------+--------------------+
        | 'Jane Doe' | 'engineer'         |
        +------------+--------------------+
        | 'Jane Doe' | 'tailor'           |
        +------------+--------------------+
        | 'Jane Doe' | 'lawyer'           |
        +------------+--------------------+
        | 'John Doe' | 'rocket scientist' |
        +------------+--------------------+
        | 'John Doe' | 'optometrist'      |
        +------------+--------------------+
        | 'John Doe' | 'chef'             |
        +------------+--------------------+
        | 'John Doe' | 'knight'           |
        +------------+--------------------+
        | 'John Doe' | 'sailor'           |
        +------------+--------------------+
    
    """
    return SplitDownView(table, field, pattern, maxsplit, flags)