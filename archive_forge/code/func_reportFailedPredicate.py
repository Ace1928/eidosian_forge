import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def reportFailedPredicate(self, recognizer, e):
    ruleName = recognizer.ruleNames[recognizer._ctx.getRuleIndex()]
    msg = 'rule ' + ruleName + ' ' + e.message
    recognizer.notifyErrorListeners(msg, e.offendingToken, e)