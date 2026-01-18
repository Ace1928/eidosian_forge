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
def list_item(self, indent):
    if self.state_machine.line[indent:]:
        indented, line_offset, blank_finish = self.state_machine.get_known_indented(indent)
    else:
        indented, indent, line_offset, blank_finish = self.state_machine.get_first_known_indented(indent)
    listitem = nodes.list_item('\n'.join(indented))
    if indented:
        self.nested_parse(indented, input_offset=line_offset, node=listitem)
    return (listitem, blank_finish)