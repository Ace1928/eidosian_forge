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
def explicit_list(self, blank_finish):
    """
        Create a nested state machine for a series of explicit markup
        constructs (including anonymous hyperlink targets).
        """
    offset = self.state_machine.line_offset + 1
    newline_offset, blank_finish = self.nested_list_parse(self.state_machine.input_lines[offset:], input_offset=self.state_machine.abs_line_offset() + 1, node=self.parent, initial_state='Explicit', blank_finish=blank_finish, match_titles=self.state_machine.match_titles)
    self.goto_line(newline_offset)
    if not blank_finish:
        self.parent += self.unindent_warning('Explicit markup')