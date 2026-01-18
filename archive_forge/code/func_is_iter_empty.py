from typing import Dict, Callable, Iterable, Optional
from .lark import Lark
from .tree import Tree, ParseTree
from .visitors import Transformer_InPlace
from .lexer import Token, PatternStr, TerminalDef
from .grammar import Terminal, NonTerminal, Symbol
from .tree_matcher import TreeMatcher, is_discarded_terminal
from .utils import is_id_continue
def is_iter_empty(i):
    try:
        _ = next(i)
        return False
    except StopIteration:
        return True