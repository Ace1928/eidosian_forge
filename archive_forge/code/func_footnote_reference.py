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
def footnote_reference(self, match, lineno):
    """
        Handles `nodes.footnote_reference` and `nodes.citation_reference`
        elements.
        """
    label = match.group('footnotelabel')
    refname = normalize_name(label)
    string = match.string
    before = string[:match.start('whole')]
    remaining = string[match.end('whole'):]
    if match.group('citationlabel'):
        refnode = nodes.citation_reference('[%s]_' % label, refname=refname)
        refnode += nodes.Text(label)
        self.document.note_citation_ref(refnode)
    else:
        refnode = nodes.footnote_reference('[%s]_' % label)
        if refname[0] == '#':
            refname = refname[1:]
            refnode['auto'] = 1
            self.document.note_autofootnote_ref(refnode)
        elif refname == '*':
            refname = ''
            refnode['auto'] = '*'
            self.document.note_symbol_footnote_ref(refnode)
        else:
            refnode += nodes.Text(label)
        if refname:
            refnode['refname'] = refname
            self.document.note_footnote_ref(refnode)
        if utils.get_trim_footnote_ref_space(self.document.settings):
            before = before.rstrip()
    return (before, [refnode], remaining, [])