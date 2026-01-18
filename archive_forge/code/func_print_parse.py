from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def print_parse(node, indent=0):
    if isinstance(node, RuleNode):
        print(' ' * (indent * 2) + str(node.rule.lhs))
        for child in node.children:
            print_parse(child, indent + 1)
    else:
        print(' ' * (indent * 2) + str(node.s))