import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def reportUnwantedToken(self, recognizer: Parser):
    if self.inErrorRecoveryMode(recognizer):
        return
    self.beginErrorCondition(recognizer)
    t = recognizer.getCurrentToken()
    tokenName = self.getTokenErrorDisplay(t)
    expecting = self.getExpectedTokens(recognizer)
    msg = 'extraneous input ' + tokenName + ' expecting ' + expecting.toString(recognizer.literalNames, recognizer.symbolicNames)
    recognizer.notifyErrorListeners(msg, t, None)