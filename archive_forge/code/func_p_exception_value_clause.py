from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def p_exception_value_clause(s, is_extern):
    """
    Parse exception value clause.

    Maps clauses to exc_check / exc_value / exc_clause as follows:
     ______________________________________________________________________
    |                             |             |             |            |
    | Clause                      | exc_check   | exc_value   | exc_clause |
    | ___________________________ | ___________ | ___________ | __________ |
    |                             |             |             |            |
    | <nothing> (default func.)   | True        | None        | False      |
    | <nothing> (cdef extern)     | False       | None        | False      |
    | noexcept                    | False       | None        | True       |
    | except <val>                | False       | <val>       | True       |
    | except? <val>               | True        | <val>       | True       |
    | except *                    | True        | None        | True       |
    | except +                    | '+'         | None        | True       |
    | except +*                   | '+'         | '*'         | True       |
    | except +<PyErr>             | '+'         | <PyErr>     | True       |
    | ___________________________ | ___________ | ___________ | __________ |

    Note that the only reason we need `exc_clause` is to raise a
    warning when `'except'` or `'noexcept'` is placed after the
    `'nogil'` keyword.
    """
    exc_clause = False
    exc_val = None
    exc_check = False if is_extern else True
    if s.sy == 'IDENT' and s.systring == 'noexcept':
        exc_clause = True
        s.next()
        exc_check = False
    elif s.sy == 'except':
        exc_clause = True
        s.next()
        if s.sy == '*':
            exc_check = True
            s.next()
        elif s.sy == '+':
            exc_check = '+'
            plus_char_pos = s.position()[2]
            s.next()
            if s.sy == 'IDENT':
                name = s.systring
                if name == 'nogil':
                    if s.position()[2] == plus_char_pos + 1:
                        error(s.position(), "'except +nogil' defines an exception handling function. Use 'except + nogil' for the 'nogil' modifier.")
                else:
                    exc_val = p_name(s, name)
                    s.next()
            elif s.sy == '*':
                exc_val = ExprNodes.CharNode(s.position(), value=u'*')
                s.next()
        else:
            if s.sy == '?':
                exc_check = True
                s.next()
            else:
                exc_check = False
            exc_val = p_test(s)
    return (exc_val, exc_check, exc_clause)