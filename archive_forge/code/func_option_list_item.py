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
def option_list_item(self, match):
    offset = self.state_machine.abs_line_offset()
    options = self.parse_option_marker(match)
    indented, indent, line_offset, blank_finish = self.state_machine.get_first_known_indented(match.end())
    if not indented:
        self.goto_line(offset)
        raise statemachine.TransitionCorrection('text')
    option_group = nodes.option_group('', *options)
    description = nodes.description('\n'.join(indented))
    option_list_item = nodes.option_list_item('', option_group, description)
    if indented:
        self.nested_parse(indented, input_offset=line_offset, node=description)
    return (option_list_item, blank_finish)