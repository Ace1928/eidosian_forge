from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
def notifyListeners(self, e: LexerNoViableAltException):
    start = self._tokenStartCharIndex
    stop = self._input.index
    text = self._input.getText(start, stop)
    msg = "token recognition error at: '" + self.getErrorDisplay(text) + "'"
    listener = self.getErrorListenerDispatch()
    listener.syntaxError(self, None, self._tokenStartLine, self._tokenStartColumn, msg, e)