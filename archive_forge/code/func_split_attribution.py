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
def split_attribution(self, indented, line_offset):
    """
        Check for a block quote attribution and split it off:

        * First line after a blank line must begin with a dash ("--", "---",
          em-dash; matches `self.attribution_pattern`).
        * Every line after that must have consistent indentation.
        * Attributions must be preceded by block quote content.

        Return a tuple of: (block quote content lines, content offset,
        attribution lines, attribution offset, remaining indented lines).
        """
    blank = None
    nonblank_seen = False
    for i in range(len(indented)):
        line = indented[i].rstrip()
        if line:
            if nonblank_seen and blank == i - 1:
                match = self.attribution_pattern.match(line)
                if match:
                    attribution_end, indent = self.check_attribution(indented, i)
                    if attribution_end:
                        a_lines = indented[i:attribution_end]
                        a_lines.trim_left(match.end(), end=1)
                        a_lines.trim_left(indent, start=1)
                        return (indented[:i], a_lines, i, indented[attribution_end:], line_offset + attribution_end)
            nonblank_seen = True
        else:
            blank = i
    else:
        return (indented, None, None, None, None)