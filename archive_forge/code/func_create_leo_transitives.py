from collections import deque
from ..tree import Tree
from ..visitors import Transformer_InPlace, v_args
from ..exceptions import UnexpectedEOF, UnexpectedToken
from ..utils import logger
from .grammar_analysis import GrammarAnalyzer
from ..grammar import NonTerminal
from .earley_common import Item, TransitiveItem
from .earley_forest import ForestSumVisitor, SymbolNode, ForestToParseTree
def create_leo_transitives(origin, start):
    visited = set()
    to_create = []
    trule = None
    previous = None
    while True:
        if origin in transitives[start]:
            previous = trule = transitives[start][origin]
            break
        is_empty_rule = not self.FIRST[origin]
        if is_empty_rule:
            break
        candidates = [candidate for candidate in columns[start] if candidate.expect is not None and origin == candidate.expect]
        if len(candidates) != 1:
            break
        originator = next(iter(candidates))
        if originator is None or originator in visited:
            break
        visited.add(originator)
        if not is_quasi_complete(originator):
            break
        trule = originator.advance()
        if originator.start != start:
            visited.clear()
        to_create.append((origin, start, originator))
        origin = originator.rule.origin
        start = originator.start
    if trule is None:
        return
    while to_create:
        origin, start, originator = to_create.pop()
        titem = None
        if previous is not None:
            titem = previous.next_titem = TransitiveItem(origin, trule, originator, previous.column)
        else:
            titem = TransitiveItem(origin, trule, originator, start)
        previous = transitives[start][origin] = titem