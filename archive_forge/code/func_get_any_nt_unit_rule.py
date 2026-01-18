from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def get_any_nt_unit_rule(g):
    """Returns a non-terminal unit rule from 'g', or None if there is none."""
    for rule in g.rules:
        if len(rule.rhs) == 1 and isinstance(rule.rhs[0], NT):
            return rule
    return None