from __future__ import absolute_import, print_function, division
import itertools
import collections
import operator
from petl.compat import next, text_type
from petl.comparison import comparable_itemgetter
from petl.util.base import Table, rowgetter, values, itervalues, \
from petl.transform.sorts import sort
def iterrecast(source, key, variablefield, valuefield, samplesize, reducers, missing):
    it = iter(source)
    try:
        hdr = next(it)
    except StopIteration:
        return
    flds = list(map(text_type, hdr))
    keyfields = key
    variablefields = variablefield
    if keyfields and (not isinstance(keyfields, (list, tuple))):
        keyfields = (keyfields,)
    if variablefields:
        if isinstance(variablefields, dict):
            pass
        elif not isinstance(variablefields, (list, tuple)):
            variablefields = (variablefields,)
    if not keyfields:
        keyfields = [f for f in flds if f not in variablefields and f != valuefield]
    if not variablefields:
        variablefields = [f for f in flds if f not in keyfields and f != valuefield]
    assert valuefield in flds, 'invalid value field: %s' % valuefield
    assert valuefield not in keyfields, 'value field cannot be keyfields'
    assert valuefield not in variablefields, 'value field cannot be variable field'
    for f in keyfields:
        assert f in flds, 'invalid keyfields field: %s' % f
    for f in variablefields:
        assert f in flds, 'invalid variable field: %s' % f
    valueindex = flds.index(valuefield)
    keyindices = [flds.index(f) for f in keyfields]
    variableindices = [flds.index(f) for f in variablefields]
    if isinstance(variablefields, dict):
        variables = variablefields
    else:
        variables = collections.defaultdict(set)
        for row in itertools.islice(it, 0, samplesize):
            for i, f in zip(variableindices, variablefields):
                variables[f].add(row[i])
        for f in variables:
            variables[f] = sorted(variables[f])
    outhdr = list(keyfields)
    for f in variablefields:
        outhdr.extend(variables[f])
    yield tuple(outhdr)
    source = sort(source, key=keyfields)
    it = itertools.islice(source, 1, None)
    getsortablekey = comparable_itemgetter(*keyindices)
    getactualkey = operator.itemgetter(*keyindices)
    groups = itertools.groupby(it, key=getsortablekey)
    for _, group in groups:
        group = list(group)
        key_value = getactualkey(group[0])
        if len(keyfields) > 1:
            out_row = list(key_value)
        else:
            out_row = [key_value]
        for f, i in zip(variablefields, variableindices):
            for variable in variables[f]:
                vals = [r[valueindex] for r in group if r[i] == variable]
                if len(vals) == 0:
                    val = missing
                elif len(vals) == 1:
                    val = vals[0]
                else:
                    if variable in reducers:
                        redu = reducers[variable]
                    else:
                        redu = list
                    val = redu(vals)
                out_row.append(val)
        yield tuple(out_row)