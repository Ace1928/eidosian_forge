from __future__ import absolute_import, print_function, division
from collections import Counter
from petl.compat import string_types, maketrans
from petl.util.base import values, Table, data, wrap
def parsecounter(table, field, parsers=(('int', int), ('float', float))):
    """
    Count the number of `str` or `unicode` values under the given fields that
    can be parsed as ints, floats or via custom parser functions. Return a
    pair of `Counter` objects, the first mapping parser names to the number of
    strings successfully parsed, the second mapping parser names to the
    number of errors. E.g.::

        >>> import petl as etl
        >>> table = [['foo', 'bar', 'baz'],
        ...          ['A', 'aaa', 2],
        ...          ['B', u'2', '3.4'],
        ...          [u'B', u'3', u'7.8', True],
        ...          ['D', '3.7', 9.0],
        ...          ['E', 42]]
        >>> counter, errors = etl.parsecounter(table, 'bar')
        >>> counter
        Counter({'float': 3, 'int': 2})
        >>> errors
        Counter({'int': 2, 'float': 1})

    The `field` argument can be a field name or index (starting from zero).

    """
    if isinstance(parsers, (list, tuple)):
        parsers = dict(parsers)
    counter, errors = (Counter(), Counter())
    for n in parsers.keys():
        counter[n] = 0
        errors[n] = 0
    for v in values(table, field):
        if isinstance(v, string_types):
            for name, parser in parsers.items():
                try:
                    parser(v)
                except:
                    errors[name] += 1
                else:
                    counter[name] += 1
    return (counter, errors)