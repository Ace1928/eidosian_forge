import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, \
def singleTokenDeletion(self, recognizer: Parser):
    nextTokenType = recognizer.getTokenStream().LA(2)
    expecting = self.getExpectedTokens(recognizer)
    if nextTokenType in expecting:
        self.reportUnwantedToken(recognizer)
        recognizer.consume()
        matchedSymbol = recognizer.getCurrentToken()
        self.reportMatch(recognizer)
        return matchedSymbol
    else:
        return None