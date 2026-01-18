from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_fieldmap_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'baz'),)
    mappings = OrderedDict()
    mappings['foo'] = 'foo'
    mappings['baz'] = ('bar', lambda v: v * 2)
    actual = fieldmap(table, mappings)
    ieq(expect, actual)