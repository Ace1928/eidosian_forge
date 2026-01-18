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
def build_regexp(definition, compile=True):
    """
    Build, compile and return a regular expression based on `definition`.

    :Parameter: `definition`: a 4-tuple (group name, prefix, suffix, parts),
        where "parts" is a list of regular expressions and/or regular
        expression definitions to be joined into an or-group.
    """
    name, prefix, suffix, parts = definition
    part_strings = []
    for part in parts:
        if type(part) is tuple:
            part_strings.append(build_regexp(part, None))
        else:
            part_strings.append(part)
    or_group = '|'.join(part_strings)
    regexp = '%(prefix)s(?P<%(name)s>%(or_group)s)%(suffix)s' % locals()
    if compile:
        return re.compile(regexp, re.UNICODE)
    else:
        return regexp