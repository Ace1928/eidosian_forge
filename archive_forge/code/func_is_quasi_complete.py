from typing import TYPE_CHECKING, Callable, Optional, List, Any
from collections import deque
from ..lexer import Token
from ..tree import Tree
from ..exceptions import UnexpectedEOF, UnexpectedToken
from ..utils import logger, OrderedSet
from .grammar_analysis import GrammarAnalyzer
from ..grammar import NonTerminal
from .earley_common import Item
from .earley_forest import ForestSumVisitor, SymbolNode, StableSymbolNode, TokenNode, ForestToParseTree
def is_quasi_complete(item):
    if item.is_complete:
        return True
    quasi = item.advance()
    while not quasi.is_complete:
        if quasi.expect not in self.NULLABLE:
            return False
        if quasi.rule.origin == start_symbol and quasi.expect == start_symbol:
            return False
        quasi = quasi.advance()
    return True