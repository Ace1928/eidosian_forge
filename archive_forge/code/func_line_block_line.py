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
def line_block_line(self, match, lineno):
    """Return one line element of a line_block."""
    indented, indent, line_offset, blank_finish = self.state_machine.get_first_known_indented(match.end(), until_blank=True)
    text = '\n'.join(indented)
    text_nodes, messages = self.inline_text(text, lineno)
    line = nodes.line(text, '', *text_nodes)
    if match.string.rstrip() != '|':
        line.indent = len(match.group(1)) - 1
    return (line, messages, blank_finish)