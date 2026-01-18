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
def inline_internal_target(self, match, lineno):
    before, inlines, remaining, sysmessages, endstring = self.inline_obj(match, lineno, self.patterns.target, nodes.target)
    if inlines and isinstance(inlines[0], nodes.target):
        assert len(inlines) == 1
        target = inlines[0]
        name = normalize_name(target.astext())
        target['names'].append(name)
        self.document.note_explicit_target(target, self.parent)
    return (before, inlines, remaining, sysmessages)