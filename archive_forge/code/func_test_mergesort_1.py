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
def test_mergesort_1():
    table1 = (('foo', 'bar'), ('A', 6), ('C', 2), ('D', 10), ('A', 9), ('F', 1))
    table2 = (('foo', 'bar'), ('B', 3), ('D', 10), ('A', 10), ('F', 4))
    expect = sort(cat(table1, table2))
    actual = mergesort(table1, table2)
    ieq(expect, actual)
    ieq(expect, actual)
    actual = mergesort(sort(table1), sort(table2), presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)