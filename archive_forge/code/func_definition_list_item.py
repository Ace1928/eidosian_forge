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
def definition_list_item(self, termline):
    indented, indent, line_offset, blank_finish = self.state_machine.get_indented()
    itemnode = nodes.definition_list_item('\n'.join(termline + list(indented)))
    lineno = self.state_machine.abs_line_number() - 1
    itemnode.source, itemnode.line = self.state_machine.get_source_and_line(lineno)
    termlist, messages = self.term(termline, lineno)
    itemnode += termlist
    definition = nodes.definition('', *messages)
    itemnode += definition
    if termline[0][-2:] == '::':
        definition += self.reporter.info('Blank line missing before literal block (after the "::")? Interpreted as a definition list item.', line=lineno + 1)
    self.nested_parse(indented, input_offset=line_offset, node=definition)
    return (itemnode, blank_finish)