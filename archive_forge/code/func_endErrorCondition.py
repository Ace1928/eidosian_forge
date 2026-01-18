import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def endErrorCondition(self, recognizer: Parser):
    self.errorRecoveryMode = False
    self.lastErrorStates = None
    self.lastErrorIndex = -1