from __future__ import absolute_import, print_function, division
from petl.util.counting import nrows
from petl.util.timing import progress, log_progress
def test_log_progress():
    table = (('foo', 'bar', 'baz'), ('a', 1, True), ('b', 2, True), ('b', 3))
    nrows(log_progress(table))