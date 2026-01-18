from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_1():
    data = [{'foo': 'a', 'bar': 1}, {'foo': 'b', 'bar': 2}, {'foo': 'c', 'bar': 2}]
    actual = fromdicts(data, header=['foo', 'bar'])
    expect = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    ieq(expect, actual)
    ieq(expect, actual)