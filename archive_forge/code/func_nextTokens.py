from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def nextTokens(self, s: ATNState, ctx: RuleContext=None):
    if ctx == None:
        return self.nextTokensNoContext(s)
    else:
        return self.nextTokensInContext(s, ctx)