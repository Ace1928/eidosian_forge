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
def handle_following_brackets(self, node, last_token, opening_bracket):
    first_child = next(cast(Callable, self._iter_children)(node))
    call_start = self._code.find_token(first_child.last_token, token.OP, opening_bracket)
    if call_start.index > last_token.index:
        last_token = call_start
    return last_token