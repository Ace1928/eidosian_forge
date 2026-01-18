from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.tree.Tree import ParseTreeListener, ParseTree, TerminalNodeImpl, ErrorNodeImpl, TerminalNode, \
def getTypedRuleContexts(self, ctxType: type):
    children = self.getChildren()
    if children is None:
        return []
    contexts = []
    for child in children:
        if not isinstance(child, ctxType):
            continue
        contexts.append(child)
    return contexts