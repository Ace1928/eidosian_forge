import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def length_or_percentage_or_unitless(argument, default=''):
    """
    Return normalized string of a length or percentage unit.

    Add <default> if there is no unit. Raise ValueError if the argument is not
    a positive measure of one of the valid CSS units (or without unit).

    >>> length_or_percentage_or_unitless('3 pt')
    '3pt'
    >>> length_or_percentage_or_unitless('3%', 'em')
    '3%'
    >>> length_or_percentage_or_unitless('3')
    '3'
    >>> length_or_percentage_or_unitless('3', 'px')
    '3px'
    """
    try:
        return get_measure(argument, length_units + ['%'])
    except ValueError:
        try:
            return get_measure(argument, ['']) + default
        except ValueError:
            return get_measure(argument, length_units + ['%'])