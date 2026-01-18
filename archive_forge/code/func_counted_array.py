import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def counted_array(expr: ParserElement, int_expr: typing.Optional[ParserElement]=None, *, intExpr: typing.Optional[ParserElement]=None) -> ParserElement:
    """Helper to define a counted list of expressions.

    This helper defines a pattern of the form::

        integer expr expr expr...

    where the leading integer tells how many expr expressions follow.
    The matched tokens returns the array of expr tokens as a list - the
    leading count token is suppressed.

    If ``int_expr`` is specified, it should be a pyparsing expression
    that produces an integer value.

    Example::

        counted_array(Word(alphas)).parse_string('2 ab cd ef')  # -> ['ab', 'cd']

        # in this parser, the leading integer value is given in binary,
        # '10' indicating that 2 values are in the array
        binary_constant = Word('01').set_parse_action(lambda t: int(t[0], 2))
        counted_array(Word(alphas), int_expr=binary_constant).parse_string('10 ab cd ef')  # -> ['ab', 'cd']

        # if other fields must be parsed after the count but before the
        # list items, give the fields results names and they will
        # be preserved in the returned ParseResults:
        count_with_metadata = integer + Word(alphas)("type")
        typed_array = counted_array(Word(alphanums), int_expr=count_with_metadata)("items")
        result = typed_array.parse_string("3 bool True True False")
        print(result.dump())

        # prints
        # ['True', 'True', 'False']
        # - items: ['True', 'True', 'False']
        # - type: 'bool'
    """
    intExpr = intExpr or int_expr
    array_expr = Forward()

    def count_field_parse_action(s, l, t):
        nonlocal array_expr
        n = t[0]
        array_expr <<= expr * n if n else Empty()
        del t[:]
    if intExpr is None:
        intExpr = Word(nums).set_parse_action(lambda t: int(t[0]))
    else:
        intExpr = intExpr.copy()
    intExpr.set_name('arrayLen')
    intExpr.add_parse_action(count_field_parse_action, call_during_try=True)
    return (intExpr + array_expr).set_name('(len) ' + str(expr) + '...')