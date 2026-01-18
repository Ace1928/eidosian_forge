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
def substitution_reference(self, match, lineno):
    before, inlines, remaining, sysmessages, endstring = self.inline_obj(match, lineno, self.patterns.substitution_ref, nodes.substitution_reference)
    if len(inlines) == 1:
        subref_node = inlines[0]
        if isinstance(subref_node, nodes.substitution_reference):
            subref_text = subref_node.astext()
            self.document.note_substitution_ref(subref_node, subref_text)
            if endstring[-1:] == '_':
                reference_node = nodes.reference('|%s%s' % (subref_text, endstring), '')
                if endstring[-2:] == '__':
                    reference_node['anonymous'] = 1
                else:
                    reference_node['refname'] = normalize_name(subref_text)
                    self.document.note_refname(reference_node)
                reference_node += subref_node
                inlines = [reference_node]
    return (before, inlines, remaining, sysmessages)