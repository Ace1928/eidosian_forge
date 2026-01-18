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
def extract_from_code(code, gettext_functions):
    """Extract strings from Python bytecode.

    >>> from genshi.template.eval import Expression
    >>> expr = Expression('_("Hello")')
    >>> list(extract_from_code(expr, GETTEXT_FUNCTIONS))
    [('_', 'Hello')]
    
    >>> expr = Expression('ngettext("You have %(num)s item", '
    ...                            '"You have %(num)s items", num)')
    >>> list(extract_from_code(expr, GETTEXT_FUNCTIONS))
    [('ngettext', ('You have %(num)s item', 'You have %(num)s items', None))]
    
    :param code: the `Code` object
    :type code: `genshi.template.eval.Code`
    :param gettext_functions: a sequence of function names
    :since: version 0.5
    """

    def _walk(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id in gettext_functions):
            strings = []

            def _add(arg):
                if isinstance(arg, _ast_Str) and isinstance(_ast_Str_value(arg), six.text_type):
                    strings.append(_ast_Str_value(arg))
                elif isinstance(arg, _ast_Str):
                    strings.append(six.text_type(_ast_Str_value(arg), 'utf-8'))
                elif arg:
                    strings.append(None)
            [_add(arg) for arg in node.args]
            if hasattr(node, 'starargs'):
                _add(node.starargs)
            if hasattr(node, 'kwargs'):
                _add(node.kwargs)
            if len(strings) == 1:
                strings = strings[0]
            else:
                strings = tuple(strings)
            yield (node.func.id, strings)
        elif node._fields:
            children = []
            for field in node._fields:
                child = getattr(node, field, None)
                if isinstance(child, list):
                    for elem in child:
                        children.append(elem)
                elif isinstance(child, ast.AST):
                    children.append(child)
            for child in children:
                for funcname, strings in _walk(child):
                    yield (funcname, strings)
    return _walk(code.ast)