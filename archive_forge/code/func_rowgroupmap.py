from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
def rowgroupmap(table, key, mapper, header=None, presorted=False, buffersize=None, tempdir=None, cache=True):
    """
    Group rows under the given key then apply `mapper` to yield zero or more
    output rows for each input group of rows.

    """
    return RowGroupMapView(table, key, mapper, header=header, presorted=presorted, buffersize=buffersize, tempdir=tempdir, cache=cache)