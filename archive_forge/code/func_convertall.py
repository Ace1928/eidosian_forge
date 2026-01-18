from __future__ import absolute_import, print_function, division
from petl.compat import next, integer_types, string_types, text_type
import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser
def convertall(table, *args, **kwargs):
    """
    Convenience function to convert all fields in the table using a common
    function or mapping. See also :func:`convert`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """
    return convert(table, fieldnames(table), *args, **kwargs)