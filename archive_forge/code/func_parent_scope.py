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
def parent_scope(node):
    while True:
        node = node.parent
        if parser_utils.is_scope(node):
            return node
        elif node.type in ('argument', 'testlist_comp'):
            if node.children[1].type in ('comp_for', 'sync_comp_for'):
                return node.children[1]
        elif node.type == 'dictorsetmaker':
            for n in node.children[1:4]:
                if n.type in ('comp_for', 'sync_comp_for'):
                    return n