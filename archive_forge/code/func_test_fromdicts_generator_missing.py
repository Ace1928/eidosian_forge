from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_generator_missing():

    def generator():
        yield OrderedDict([('foo', 'a'), ('bar', 1)])
        yield OrderedDict([('foo', 'b'), ('bar', 2)])
        yield OrderedDict([('foo', 'c'), ('baz', 2)])
    actual = fromdicts(generator(), missing='x')
    expect = (('foo', 'bar', 'baz'), ('a', 1, 'x'), ('b', 2, 'x'), ('c', 'x', 2))
    ieq(expect, actual)