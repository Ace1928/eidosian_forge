from io import StringIO
import sys
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException
def getErrorDisplayForChar(self, c: str):
    if ord(c[0]) == Token.EOF:
        return '<EOF>'
    elif c == '\n':
        return '\\n'
    elif c == '\t':
        return '\\t'
    elif c == '\r':
        return '\\r'
    else:
        return c