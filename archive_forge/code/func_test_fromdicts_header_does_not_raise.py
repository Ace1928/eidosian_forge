from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_header_does_not_raise():
    data = [{'foo': 'a', 'bar': 1}, {'foo': 'b', 'bar': 2}, {'foo': 'c', 'bar': 2}]
    actual = fromdicts(data)
    assert actual.header()