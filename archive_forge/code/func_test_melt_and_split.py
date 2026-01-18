from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_melt_and_split():
    table = (('id', 'parad0', 'parad1', 'parad2', 'tempd0', 'tempd1', 'tempd2'), ('1', '12', '34', '56', '37.2', '37.4', '37.9'), ('2', '23', '45', '67', '37.1', '37.8', '36.9'))
    expectation = (('id', 'value', 'variable', 'day'), ('1', '12', 'para', '0'), ('1', '34', 'para', '1'), ('1', '56', 'para', '2'), ('1', '37.2', 'temp', '0'), ('1', '37.4', 'temp', '1'), ('1', '37.9', 'temp', '2'), ('2', '23', 'para', '0'), ('2', '45', 'para', '1'), ('2', '67', 'para', '2'), ('2', '37.1', 'temp', '0'), ('2', '37.8', 'temp', '1'), ('2', '36.9', 'temp', '2'))
    step1 = melt(table, key='id')
    step2 = split(step1, 'variable', 'd', ('variable', 'day'))
    ieq(expectation, step2)