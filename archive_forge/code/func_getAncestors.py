from io import StringIO
from antlr4.Token import Token
from antlr4.Utils import escapeWhitespace
from antlr4.tree.Tree import RuleNode, ErrorNode, TerminalNode, Tree, ParseTree
@classmethod
def getAncestors(cls, t: Tree):
    ancestors = []
    t = t.getParent()
    while t is not None:
        ancestors.insert(0, t)
        t = t.getParent()
    return ancestors