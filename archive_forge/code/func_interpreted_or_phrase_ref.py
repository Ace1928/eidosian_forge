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
def interpreted_or_phrase_ref(self, match, lineno):
    end_pattern = self.patterns.interpreted_or_phrase_ref
    string = match.string
    matchstart = match.start('backquote')
    matchend = match.end('backquote')
    rolestart = match.start('role')
    role = match.group('role')
    position = ''
    if role:
        role = role[1:-1]
        position = 'prefix'
    elif self.quoted_start(match):
        return (string[:matchend], [], string[matchend:], [])
    endmatch = end_pattern.search(string[matchend:])
    if endmatch and endmatch.start(1):
        textend = matchend + endmatch.end()
        if endmatch.group('role'):
            if role:
                msg = self.reporter.warning('Multiple roles in interpreted text (both prefix and suffix present; only one allowed).', line=lineno)
                text = unescape(string[rolestart:textend], True)
                prb = self.problematic(text, text, msg)
                return (string[:rolestart], [prb], string[textend:], [msg])
            role = endmatch.group('suffix')[1:-1]
            position = 'suffix'
        escaped = endmatch.string[:endmatch.start(1)]
        rawsource = unescape(string[matchstart:textend], True)
        if rawsource[-1:] == '_':
            if role:
                msg = self.reporter.warning('Mismatch: both interpreted text role %s and reference suffix.' % position, line=lineno)
                text = unescape(string[rolestart:textend], True)
                prb = self.problematic(text, text, msg)
                return (string[:rolestart], [prb], string[textend:], [msg])
            return self.phrase_ref(string[:matchstart], string[textend:], rawsource, escaped, unescape(escaped))
        else:
            rawsource = unescape(string[rolestart:textend], True)
            nodelist, messages = self.interpreted(rawsource, escaped, role, lineno)
            return (string[:rolestart], nodelist, string[textend:], messages)
    msg = self.reporter.warning('Inline interpreted text or phrase reference start-string without end-string.', line=lineno)
    text = unescape(string[matchstart:matchend], True)
    prb = self.problematic(text, text, msg)
    return (string[:matchstart], [prb], string[matchend:], [msg])