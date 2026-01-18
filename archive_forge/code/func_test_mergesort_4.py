from __future__ import absolute_import, print_function, division
import os
import gc
import logging
from datetime import datetime
import platform
import pytest
from petl.compat import next
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.util import nrows
from petl.transform.basics import cat
from petl.transform.sorts import sort, mergesort, issorted
def test_mergesort_4():
    table1 = (('foo', 'bar', 'baz'), (1, 'A', True), (2, 'B', None), (4, 'C', True))
    table2 = (('bar', 'baz', 'quux'), ('A', True, 42.0), ('B', False, 79.3), ('C', False, 12.4))
    expect = sort(cat(table1, table2), key='bar')
    actual = mergesort(table1, table2, key='bar')
    ieq(expect, actual)
    ieq(expect, actual)