import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def reportInputMismatch(self, recognizer: Parser, e: InputMismatchException):
    msg = 'mismatched input ' + self.getTokenErrorDisplay(e.offendingToken) + ' expecting ' + e.getExpectedTokens().toString(recognizer.literalNames, recognizer.symbolicNames)
    recognizer.notifyErrorListeners(msg, e.offendingToken, e)