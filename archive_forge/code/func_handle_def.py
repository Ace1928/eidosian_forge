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
def handle_def(self, node, first_token, last_token):
    if not node.body and (getattr(node, 'doc_node', None) or getattr(node, 'doc', None)):
        last_token = self._code.find_token(last_token, token.STRING)
    if first_token.index > 0:
        prev = self._code.prev_token(first_token)
        if util.match_token(prev, token.OP, '@'):
            first_token = prev
    return (first_token, last_token)