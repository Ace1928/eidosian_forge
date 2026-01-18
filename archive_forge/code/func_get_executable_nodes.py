import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def get_executable_nodes(node, last_added=False):
    """
    For static analysis.
    """
    result = []
    typ = node.type
    if typ == 'name':
        next_leaf = node.get_next_leaf()
        if last_added is False and node.parent.type != 'param' and (next_leaf != '='):
            result.append(node)
    elif typ == 'expr_stmt':
        result.append(node)
        for child in node.children:
            result += get_executable_nodes(child, last_added=True)
    elif typ == 'decorator':
        if node.children[-2] == ')':
            node = node.children[-3]
            if node != '(':
                result += get_executable_nodes(node)
    else:
        try:
            children = node.children
        except AttributeError:
            pass
        else:
            if node.type in _EXECUTE_NODES and (not last_added):
                result.append(node)
            for child in children:
                result += get_executable_nodes(child, last_added)
    return result