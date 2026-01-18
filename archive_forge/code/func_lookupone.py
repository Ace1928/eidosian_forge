from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.errors import DuplicateKeyError
from petl.util.base import Table, asindices, asdict, Record, rowgetter
def lookupone(table, key, value=None, dictionary=None, strict=False):
    """
    Load a dictionary with data from the given table, assuming there is
    at most one value for each key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['a', 1],
        ...           ['b', 2],
        ...           ['b', 3]]
        >>> # if the specified key is not unique and strict=False (default),
        ... # the first value wins
        ... lkp = etl.lookupone(table1, 'foo', 'bar')
        >>> lkp['a']
        1
        >>> lkp['b']
        2
        >>> # if the specified key is not unique and strict=True, will raise
        ... # DuplicateKeyError
        ... try:
        ...     lkp = etl.lookupone(table1, 'foo', strict=True)
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
        >>> lkp = etl.lookupone(table2, ('foo', 'bar'), 'baz')
        >>> lkp[('a', 1)]
        True
        >>> lkp[('b', 2)]
        False
        >>> lkp[('b', 3)]
        True
        >>> # data can be loaded into an existing dictionary-like
        ... # object, including persistent dictionaries created via the
        ... # shelve module
        ... import shelve
        >>> lkp = shelve.open('example.dat', flag='n')
        >>> lkp = etl.lookupone(table1, 'foo', 'bar', lkp)
        >>> lkp.close()
        >>> lkp = shelve.open('example.dat', flag='r')
        >>> lkp['a']
        1
        >>> lkp['b']
        2

    """
    if dictionary is None:
        dictionary = dict()
    it, getkey, getvalue = _setup_lookup(table, key, value)
    for row in it:
        k = getkey(row)
        if strict and k in dictionary:
            raise DuplicateKeyError(k)
        elif k not in dictionary:
            v = getvalue(row)
            dictionary[k] = v
    return dictionary