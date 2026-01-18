from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_fromdicts_generator_random_access():

    def generator():
        for i in range(5):
            yield OrderedDict([('n', i), ('foo', 100 * i), ('bar', 200 * i)])
    actual = fromdicts(generator(), sample=3)
    assert actual.header() == ('n', 'foo', 'bar')
    it1 = iter(actual)
    first_row1 = next(it1)
    first_row2 = next(it1)
    it2 = iter(actual)
    second_row1 = next(it2)
    second_row2 = next(it2)
    assert first_row1 == second_row1
    assert first_row2 == second_row2
    second_row3 = next(it2)
    first_row3 = next(it1)
    assert second_row3 == first_row3
    ieq(actual, actual)
    assert actual.header() == ('n', 'foo', 'bar')
    assert len(actual) == 6