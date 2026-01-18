from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def readLexerActions(self, atn: ATN):
    if atn.grammarType == ATNType.LEXER:
        count = self.readInt()
        atn.lexerActions = [None] * count
        for i in range(0, count):
            actionType = self.readInt()
            data1 = self.readInt()
            data2 = self.readInt()
            lexerAction = self.lexerActionFactory(actionType, data1, data2)
            atn.lexerActions[i] = lexerAction