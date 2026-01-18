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
def create_value(self, node):
    from jedi.inference import value
    if node == self.tree_node:
        assert self.is_module()
        return self.get_value()
    parent_context = self.create_context(node)
    if node.type in ('funcdef', 'lambdef'):
        func = value.FunctionValue.from_context(parent_context, node)
        if parent_context.is_class():
            class_value = parent_context.parent_context.create_value(parent_context.tree_node)
            instance = value.AnonymousInstance(self.inference_state, parent_context.parent_context, class_value)
            func = value.BoundMethod(instance=instance, class_context=class_value.as_context(), function=func)
        return func
    elif node.type == 'classdef':
        return value.ClassValue(self.inference_state, parent_context, node)
    else:
        raise NotImplementedError("Probably shouldn't happen: %s" % node)