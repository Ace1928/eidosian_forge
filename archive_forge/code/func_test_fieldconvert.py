from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
from functools import partial
def test_fieldconvert():
    table1 = (('foo', 'bar', 'baz'), ('A', 1, 2), ('B', '2', '3.4'), (u'B', u'3', u'7.8', True), ('D', 'xyz', 9.0), ('E', None))
    converters = {'foo': str, 'bar': int, 'baz': float}
    table5 = convert(table1, converters, errorvalue='error')
    expect5 = (('foo', 'bar', 'baz'), ('A', 1, 2.0), ('B', 2, 3.4), ('B', 3, 7.8, True), ('D', 'error', 9.0), ('E', 'error'))
    ieq(expect5, table5)
    table6 = convert(table1, errorvalue='err')
    table6['foo'] = str
    table6['bar'] = int
    table6['baz'] = float
    expect6 = (('foo', 'bar', 'baz'), ('A', 1, 2.0), ('B', 2, 3.4), ('B', 3, 7.8, True), ('D', 'err', 9.0), ('E', 'err'))
    ieq(expect6, table6)
    table7 = convert(table1)
    table7['foo'] = ('replace', 'B', 'BB')
    expect7 = (('foo', 'bar', 'baz'), ('A', 1, 2), ('BB', '2', '3.4'), (u'BB', u'3', u'7.8', True), ('D', 'xyz', 9.0), ('E', None))
    ieq(expect7, table7)
    converters = [str, int, float]
    table8 = convert(table1, converters, errorvalue='error')
    expect8 = (('foo', 'bar', 'baz'), ('A', 1, 2.0), ('B', 2, 3.4), ('B', 3, 7.8, True), ('D', 'error', 9.0), ('E', 'error'))
    ieq(expect8, table8)
    converters = [str, None, float]
    table9 = convert(table1, converters, errorvalue='error')
    expect9 = (('foo', 'bar', 'baz'), ('A', 1, 2.0), ('B', '2', 3.4), ('B', u'3', 7.8, True), ('D', 'xyz', 9.0), ('E', None))
    ieq(expect9, table9)