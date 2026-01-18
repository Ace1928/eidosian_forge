from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def edgeFactory(self, atn: ATN, type: int, src: int, trg: int, arg1: int, arg2: int, arg3: int, sets: list):
    target = atn.states[trg]
    if type > len(self.edgeFactories) or self.edgeFactories[type] is None:
        raise Exception('The specified transition type: ' + str(type) + ' is not valid.')
    else:
        return self.edgeFactories[type](atn, src, trg, arg1, arg2, arg3, sets, target)