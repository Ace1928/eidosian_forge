from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
def replaceall(table, a, b, **kwargs):
    """
    Convenience function to replace all instances of `a` with `b` under all
    fields. See also :func:`convertall`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """
    return convertall(table, {a: b}, **kwargs)