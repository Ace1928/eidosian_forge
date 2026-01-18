from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import DuplicateKeyError, FieldSelectionError
from petl.test.helpers import eq_
from petl import cut, lookup, lookupone, dictlookup, dictlookupone, \
def test_dictlookupone():
    t1 = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 3))
    try:
        dictlookupone(t1, 'foo', strict=True)
    except DuplicateKeyError:
        pass
    else:
        assert False, 'expected error'
    actual = dictlookupone(t1, 'foo', strict=False)
    expect = {'a': {'foo': 'a', 'bar': 1}, 'b': {'foo': 'b', 'bar': 2}}
    eq_(expect, actual)
    actual = dictlookupone(cut(t1, 'foo'), 'foo')
    expect = {'a': {'foo': 'a'}, 'b': {'foo': 'b'}}
    eq_(expect, actual)
    t2 = (('foo', 'bar', 'baz'), ('a', 1, True), ('b', 2, False), ('b', 3, True), ('b', 3, False))
    actual = dictlookupone(t2, ('foo', 'bar'), strict=False)
    expect = {('a', 1): {'foo': 'a', 'bar': 1, 'baz': True}, ('b', 2): {'foo': 'b', 'bar': 2, 'baz': False}, ('b', 3): {'foo': 'b', 'bar': 3, 'baz': True}}
    eq_(expect, actual)