import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def getErrorRecoverySet(self, recognizer: Parser):
    atn = recognizer._interp.atn
    ctx = recognizer._ctx
    recoverSet = IntervalSet()
    while ctx is not None and ctx.invokingState >= 0:
        invokingState = atn.states[ctx.invokingState]
        rt = invokingState.transitions[0]
        follow = atn.nextTokens(rt.followState)
        recoverSet.addSet(follow)
        ctx = ctx.parentCtx
    recoverSet.removeOne(Token.EPSILON)
    return recoverSet