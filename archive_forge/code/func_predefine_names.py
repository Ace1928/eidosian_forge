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
@contextmanager
def predefine_names(self, flow_scope, dct):
    predefined = self.predefined_names
    predefined[flow_scope] = dct
    try:
        yield
    finally:
        del predefined[flow_scope]