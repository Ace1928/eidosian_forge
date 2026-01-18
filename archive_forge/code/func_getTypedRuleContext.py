from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.tree.Tree import ParseTreeListener, ParseTree, TerminalNodeImpl, ErrorNodeImpl, TerminalNode, \
def getTypedRuleContext(self, ctxType: type, i: int):
    return self.getChild(i, ctxType)