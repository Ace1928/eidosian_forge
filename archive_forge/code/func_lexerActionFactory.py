from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def lexerActionFactory(self, type: int, data1: int, data2: int):
    if type > len(self.actionFactories) or self.actionFactories[type] is None:
        raise Exception('The specified lexer action type ' + str(type) + ' is not valid.')
    else:
        return self.actionFactories[type](data1, data2)