from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromjson_3():
    f = NamedTemporaryFile(delete=False, mode='w')
    data = '[{"foo": "a", "bar": 1}, {"foo": "b"}, {"foo": "c", "bar": 2, "baz": true}]'
    f.write(data)
    f.close()
    actual = fromjson(f.name, header=['foo', 'bar'])
    expect = (('foo', 'bar'), ('a', 1), ('b', None), ('c', 2))
    ieq(expect, actual)
    ieq(expect, actual)