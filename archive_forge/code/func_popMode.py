from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
def popMode(self):
    if len(self._modeStack) == 0:
        raise Exception('Empty Stack')
    if self._interp.debug:
        print('popMode back to ' + self._modeStack[:-1], file=self._output)
    self.mode(self._modeStack.pop())
    return self._mode