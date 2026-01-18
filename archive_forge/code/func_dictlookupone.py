from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.errors import DuplicateKeyError
from petl.util.base import Table, asindices, asdict, Record, rowgetter
def dictlookupone(table, key, dictionary=None, strict=False):
    """
    Load a dictionary with data from the given table, mapping to dicts,
    assuming there is at most one row for each key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['b', 3]]
        >>> # if the specified key is not unique and strict=False (default),
        ... # the first value wins
        ... lkp = etl.dictlookupone(table1, 'foo')
        >>> lkp['a']
        {'foo': 'a', 'bar': 1}
        >>> lkp['b']
        {'foo': 'b', 'bar': 2}
        >>> # if the specified key is not unique and strict=True, will raise
        ... # DuplicateKeyError
        ... try:
        ...     lkp = etl.dictlookupone(table1, 'foo', strict=True)
        ... except etl.errors.DuplicateKeyError as e:
        ...     print(e)
        ...
        duplicate key: 'b'
        >>> # compound keys are supported
        ... table2 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, True],
        ...           ['b', 2, False],
        ...           ['b', 3, True],
        ...           ['b', 3, False]]
        >>> lkp = etl.dictlookupone(table2, ('foo', 'bar'))
        >>> lkp[('a', 1)]
        {'foo': 'a', 'bar': 1, 'baz': True}
        >>> lkp[('b', 2)]
        {'foo': 'b', 'bar': 2, 'baz': False}
        >>> lkp[('b', 3)]
        {'foo': 'b', 'bar': 3, 'baz': True}
        >>> # data can be loaded into an existing dictionary-like
        ... # object, including persistent dictionaries created via the
        ... # shelve module
        ... import shelve
        >>> lkp = shelve.open('example.dat', flag='n')
        >>> lkp = etl.dictlookupone(table1, 'foo', lkp)
        >>> lkp.close()
        >>> lkp = shelve.open('example.dat', flag='r')
        >>> lkp['a']
        {'foo': 'a', 'bar': 1}
        >>> lkp['b']
        {'foo': 'b', 'bar': 2}

    """
    if dictionary is None:
        dictionary = dict()
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        hdr = []
    flds = list(map(text_type, hdr))
    keyindices = asindices(hdr, key)
    assert len(keyindices) > 0, 'no key selected'
    getkey = operator.itemgetter(*keyindices)
    for row in it:
        k = getkey(row)
        if strict and k in dictionary:
            raise DuplicateKeyError(k)
        elif k not in dictionary:
            d = asdict(flds, row)
            dictionary[k] = d
    return dictionary