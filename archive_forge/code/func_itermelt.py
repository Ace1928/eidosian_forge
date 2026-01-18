from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
def itermelt(source, key, variables, variablefield, valuefield):
    if key is None and variables is None:
        raise ValueError('either key or variables must be specified')
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    key_indices = variables_indices = None
    if key is not None:
        key_indices = asindices(hdr, key)
    if variables is not None:
        if not isinstance(variables, (list, tuple)):
            variables = (variables,)
        variables_indices = asindices(hdr, variables)
    if key is None:
        key_indices = [i for i in range(len(hdr)) if i not in variables_indices]
    if variables is None:
        variables_indices = [i for i in range(len(hdr)) if i not in key_indices]
        variables = [hdr[i] for i in variables_indices]
    getkey = rowgetter(*key_indices)
    outhdr = [hdr[i] for i in key_indices]
    outhdr.append(variablefield)
    outhdr.append(valuefield)
    yield tuple(outhdr)
    for row in it:
        k = getkey(row)
        for v, i in zip(variables, variables_indices):
            try:
                o = list(k)
                o.append(v)
                o.append(row[i])
                yield tuple(o)
            except IndexError:
                pass