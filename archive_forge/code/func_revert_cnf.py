from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def revert_cnf(node):
    """Reverts a parse tree (RuleNode) to its original non-CNF form (Node)."""
    if isinstance(node, T):
        return node
    if node.rule.lhs.name.startswith('__T_'):
        return node.children[0]
    else:
        children = []
        for child in map(revert_cnf, node.children):
            if isinstance(child, RuleNode) and child.rule.lhs.name.startswith('__SP_'):
                children += child.children
            else:
                children.append(child)
        if isinstance(node.rule, UnitSkipRule):
            return unroll_unit_skiprule(node.rule.lhs, node.rule.rhs, node.rule.skipped_rules, children, node.rule.weight, node.rule.alias)
        else:
            return RuleNode(node.rule, children)