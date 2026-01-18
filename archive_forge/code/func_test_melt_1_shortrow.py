from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_melt_1_shortrow():
    table = (('id', 'gender', 'age'), (1, 'F', 12), (2, 'M', 17), (3, 'M'), (4,))
    expectation = (('id', 'variable', 'value'), (1, 'gender', 'F'), (1, 'age', 12), (2, 'gender', 'M'), (2, 'age', 17), (3, 'gender', 'M'))
    result = melt(table, key='id')
    ieq(expectation, result)
    result = melt(table, key='id', variablefield='variable', valuefield='value')
    ieq(expectation, result)