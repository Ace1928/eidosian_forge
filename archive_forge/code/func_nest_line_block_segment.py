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
def nest_line_block_segment(self, block):
    indents = [item.indent for item in block]
    least = min(indents)
    new_items = []
    new_block = nodes.line_block()
    for item in block:
        if item.indent > least:
            new_block.append(item)
        else:
            if len(new_block):
                self.nest_line_block_segment(new_block)
                new_items.append(new_block)
                new_block = nodes.line_block()
            new_items.append(item)
    if len(new_block):
        self.nest_line_block_segment(new_block)
        new_items.append(new_block)
    block[:] = new_items