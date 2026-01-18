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
def malformed_table(self, block, detail='', offset=0):
    block.replace(self.double_width_pad_char, '')
    data = '\n'.join(block)
    message = 'Malformed table.'
    startline = self.state_machine.abs_line_number() - len(block) + 1
    if detail:
        message += '\n' + detail
    error = self.reporter.error(message, nodes.literal_block(data, data), line=startline + offset)
    return [error]