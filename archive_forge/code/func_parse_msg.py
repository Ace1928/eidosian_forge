from gettext import NullTranslations
import os
import re
from functools import partial
from types import FunctionType
import six
from genshi.core import Attrs, Namespace, QName, START, END, TEXT, \
from genshi.template.base import DirectiveFactory, EXPR, SUB, _apply_directives
from genshi.template.directives import Directive, StripDirective
from genshi.template.markup import MarkupTemplate, EXEC
from genshi.compat import ast, IS_PYTHON2, _ast_Str, _ast_Str_value
def parse_msg(string, regex=re.compile('(?:\\[(\\d+)\\:)|(?<!\\\\)\\]')):
    """Parse a translated message using Genshi mixed content message
    formatting.

    >>> parse_msg("See [1:Help].")
    [(0, 'See '), (1, 'Help'), (0, '.')]

    >>> parse_msg("See [1:our [2:Help] page] for details.")
    [(0, 'See '), (1, 'our '), (2, 'Help'), (1, ' page'), (0, ' for details.')]

    >>> parse_msg("[2:Details] finden Sie in [1:Hilfe].")
    [(2, 'Details'), (0, ' finden Sie in '), (1, 'Hilfe'), (0, '.')]

    >>> parse_msg("[1:] Bilder pro Seite anzeigen.")
    [(1, ''), (0, ' Bilder pro Seite anzeigen.')]

    :param string: the translated message string
    :return: a list of ``(order, string)`` tuples
    :rtype: `list`
    """
    parts = []
    stack = [0]
    while True:
        mo = regex.search(string)
        if not mo:
            break
        if mo.start() or stack[-1]:
            parts.append((stack[-1], string[:mo.start()]))
        string = string[mo.end():]
        orderno = mo.group(1)
        if orderno is not None:
            stack.append(int(orderno))
        else:
            stack.pop()
        if not stack:
            break
    if string:
        parts.append((stack[-1], string))
    return parts