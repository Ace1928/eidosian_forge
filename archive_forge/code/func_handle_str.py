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
def handle_str(self, first_token, last_token):
    last = self._code.next_token(last_token)
    while util.match_token(last, token.STRING):
        last_token = last
        last = self._code.next_token(last_token)
    return (first_token, last_token)