from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from itertools import islice
from petl.compat import izip_longest, text_type, next
from petl.util.base import asindices, Table
def facetcolumns(table, key, missing=None):
    """
    Like :func:`petl.util.materialise.columns` but stratified by values of the
    given key field. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          ['a', 1, True],
        ...          ['b', 2, True],
        ...          ['b', 3]]
        >>> fc = etl.facetcolumns(table, 'foo')
        >>> fc['a']
        {'foo': ['a'], 'bar': [1], 'baz': [True]}
        >>> fc['b']
        {'foo': ['b', 'b'], 'bar': [2, 3], 'baz': [True, None]}

    """
    fct = dict()
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    indices = asindices(hdr, key)
    assert len(indices) > 0, 'no key field selected'
    getkey = operator.itemgetter(*indices)
    for row in it:
        kv = getkey(row)
        if kv not in fct:
            cols = dict()
            for f in flds:
                cols[f] = list()
            fct[kv] = cols
        else:
            cols = fct[kv]
        for f, v in izip_longest(flds, row, fillvalue=missing):
            if f in cols:
                cols[f].append(v)
    return fct