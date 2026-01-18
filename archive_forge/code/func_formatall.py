from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
def formatall(table, fmt, **kwargs):
    """
    Convenience function to format all values in all fields using the
    `fmt` format string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """
    conv = lambda v: fmt.format(v)
    return convertall(table, conv, **kwargs)