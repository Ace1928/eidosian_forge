import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def get_following_comment_same_line(node):
    """
    returns (as string) any comment that appears on the same line,
    after the node, including the #
    """
    try:
        if node.type == 'for_stmt':
            whitespace = node.children[5].get_first_leaf().prefix
        elif node.type == 'with_stmt':
            whitespace = node.children[3].get_first_leaf().prefix
        elif node.type == 'funcdef':
            whitespace = node.children[4].get_first_leaf().get_next_leaf().prefix
        else:
            whitespace = node.get_last_leaf().get_next_leaf().prefix
    except AttributeError:
        return None
    except ValueError:
        return None
    if '#' not in whitespace:
        return None
    comment = whitespace[whitespace.index('#'):]
    if '\r' in comment:
        comment = comment[:comment.index('\r')]
    if '\n' in comment:
        comment = comment[:comment.index('\n')]
    return comment