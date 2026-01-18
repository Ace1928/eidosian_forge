from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
def dictconverter(d):

    def conv(v):
        try:
            if v in d:
                return d[v]
            else:
                return v
        except TypeError:
            return v
    return conv