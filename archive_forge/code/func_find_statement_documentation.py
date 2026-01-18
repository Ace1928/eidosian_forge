import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def find_statement_documentation(tree_node):
    if tree_node.type == 'expr_stmt':
        tree_node = tree_node.parent
        maybe_string = tree_node.get_next_sibling()
        if maybe_string is not None:
            if maybe_string.type == 'simple_stmt':
                maybe_string = maybe_string.children[0]
                if maybe_string.type == 'string':
                    return cleandoc(safe_literal_eval(maybe_string.value))
    return ''