import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
from parso.python.parser import Parser
from parso.python import tree
from jedi.inference.base_value import NO_VALUES
from jedi.inference.syntax_tree import infer_atom
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.compiled import get_string_value_set
from jedi.cache import signature_time_cache, memoize_method
from jedi.parser_utils import get_parent_scope
def get_signature_details(module, position):
    leaf = module.get_leaf_for_position(position, include_prefixes=True)
    if leaf.start_pos >= position:
        leaf = leaf.get_previous_leaf()
        if leaf is None:
            return None
    node = leaf.parent
    while node is not None:
        if node.type in ('funcdef', 'classdef', 'decorated', 'async_stmt'):
            return None
        additional_children = []
        for n in reversed(node.children):
            if n.start_pos < position:
                if n.type == 'error_node':
                    result = _get_signature_details_from_error_node(n, additional_children, position)
                    if result is not None:
                        return result
                    additional_children[0:0] = n.children
                    continue
                additional_children.insert(0, n)
        if node.type == 'trailer' and node.children[0] == '(' or (node.type == 'decorator' and node.children[2] == '('):
            if not (leaf is node.children[-1] and position >= leaf.end_pos):
                leaf = node.get_previous_leaf()
                if leaf is None:
                    return None
                return CallDetails(node.children[0] if node.type == 'trailer' else node.children[2], node.children, position)
        node = node.parent
    return None