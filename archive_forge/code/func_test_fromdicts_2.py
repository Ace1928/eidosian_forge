from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_2():
    data = [{'foo': 'a', 'bar': 1}, {'foo': 'b'}, {'foo': 'c', 'bar': 2, 'baz': True}]
    actual = fromdicts(data, header=['bar', 'baz', 'foo'])
    expect = (('bar', 'baz', 'foo'), (1, None, 'a'), (None, None, 'b'), (2, True, 'c'))
    ieq(expect, actual)
    ieq(expect, actual)