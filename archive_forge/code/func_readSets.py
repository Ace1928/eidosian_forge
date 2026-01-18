from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def readSets(self, atn: ATN, sets: list):
    m = self.readInt()
    for i in range(0, m):
        iset = IntervalSet()
        sets.append(iset)
        n = self.readInt()
        containsEof = self.readInt()
        if containsEof != 0:
            iset.addOne(-1)
        for j in range(0, n):
            i1 = self.readInt()
            i2 = self.readInt()
            iset.addRange(range(i1, i2 + 1))