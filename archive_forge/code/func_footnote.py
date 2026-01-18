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
def footnote(self, match):
    src, srcline = self.state_machine.get_source_and_line()
    indented, indent, offset, blank_finish = self.state_machine.get_first_known_indented(match.end())
    label = match.group(1)
    name = normalize_name(label)
    footnote = nodes.footnote('\n'.join(indented))
    footnote.source = src
    footnote.line = srcline
    if name[0] == '#':
        name = name[1:]
        footnote['auto'] = 1
        if name:
            footnote['names'].append(name)
        self.document.note_autofootnote(footnote)
    elif name == '*':
        name = ''
        footnote['auto'] = '*'
        self.document.note_symbol_footnote(footnote)
    else:
        footnote += nodes.label('', label)
        footnote['names'].append(name)
        self.document.note_footnote(footnote)
    if name:
        self.document.note_explicit_target(footnote, footnote)
    else:
        self.document.set_id(footnote, footnote)
    if indented:
        self.nested_parse(indented, input_offset=offset, node=footnote)
    return ([footnote], blank_finish)