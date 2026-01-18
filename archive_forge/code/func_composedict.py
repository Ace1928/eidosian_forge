from __future__ import absolute_import, print_function, division
import operator
from collections import OrderedDict
from petl.compat import next, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError
from petl.util.base import Table, expr, rowgroupby, Record
from petl.transform.sorts import sort
def composedict(d, srcfld):

    def g(rec):
        k = rec[srcfld]
        if k in d:
            return d[k]
        else:
            return k
    return g