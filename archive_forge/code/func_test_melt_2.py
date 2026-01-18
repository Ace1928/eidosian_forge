from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_melt_2():
    table = (('id', 'time', 'height', 'weight'), (1, 11, 66.4, 12.2), (2, 16, 53.2, 17.3), (3, 12, 34.5, 9.4))
    expectation = (('id', 'time', 'variable', 'value'), (1, 11, 'height', 66.4), (1, 11, 'weight', 12.2), (2, 16, 'height', 53.2), (2, 16, 'weight', 17.3), (3, 12, 'height', 34.5), (3, 12, 'weight', 9.4))
    result = melt(table, key=('id', 'time'))
    ieq(expectation, result)
    expectation = (('id', 'time', 'variable', 'value'), (1, 11, 'height', 66.4), (2, 16, 'height', 53.2), (3, 12, 'height', 34.5))
    result = melt(table, key=('id', 'time'), variables='height')
    print(result)
    ieq(expectation, result)