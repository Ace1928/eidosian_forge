import ast
import collections
import io
import sys
import token
import tokenize
from abc import ABCMeta
from ast import Module, expr, AST
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, Any, TYPE_CHECKING
from six import iteritems
def is_non_coding_token(token_type):
    """
    These are considered non-coding tokens, as they don't affect the syntax tree.
    """
    return token_type >= token.N_TOKENS