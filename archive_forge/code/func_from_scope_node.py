from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, \
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils
def from_scope_node(scope_node, is_nested=True):
    if scope_node == self.tree_node:
        return self
    if scope_node.type in ('funcdef', 'lambdef', 'classdef'):
        return self.create_value(scope_node).as_context()
    elif scope_node.type in ('comp_for', 'sync_comp_for'):
        parent_context = from_scope_node(parent_scope(scope_node.parent))
        if node.start_pos >= scope_node.children[-1].start_pos:
            return parent_context
        return CompForContext(parent_context, scope_node)
    raise Exception("There's a scope that was not managed: %s" % scope_node)