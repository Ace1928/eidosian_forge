import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def make_enumerator(self, ordinal, sequence, format):
    """
        Construct and return the next enumerated list item marker, and an
        auto-enumerator ("#" instead of the regular enumerator).

        Return ``None`` for invalid (out of range) ordinals.
        """
    if sequence == '#':
        enumerator = '#'
    elif sequence == 'arabic':
        enumerator = str(ordinal)
    else:
        if sequence.endswith('alpha'):
            if ordinal > 26:
                return None
            enumerator = chr(ordinal + ord('a') - 1)
        elif sequence.endswith('roman'):
            try:
                enumerator = roman.toRoman(ordinal)
            except roman.RomanError:
                return None
        else:
            raise ParserError('unknown enumerator sequence: "%s"' % sequence)
        if sequence.startswith('lower'):
            enumerator = enumerator.lower()
        elif sequence.startswith('upper'):
            enumerator = enumerator.upper()
        else:
            raise ParserError('unknown enumerator sequence: "%s"' % sequence)
    formatinfo = self.enum.formatinfo[format]
    next_enumerator = formatinfo.prefix + enumerator + formatinfo.suffix + ' '
    auto_enumerator = formatinfo.prefix + '#' + formatinfo.suffix + ' '
    return (next_enumerator, auto_enumerator)