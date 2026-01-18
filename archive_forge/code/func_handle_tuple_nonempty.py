import ast
import numbers
import sys
import token
from ast import Module
from typing import Callable, List, Union, cast, Optional, Tuple, TYPE_CHECKING
import six
from . import util
from .asttokens import ASTTokens
from .util import AstConstant
from .astroid_compat import astroid_node_classes as nc, BaseContainer as AstroidBaseContainer
def handle_tuple_nonempty(self, node, first_token, last_token):
    first_token, last_token = self.handle_bare_tuple(node, first_token, last_token)
    return self._gobble_parens(first_token, last_token, False)