from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def readModes(self, atn: ATN):
    nmodes = self.readInt()
    for i in range(0, nmodes):
        s = self.readInt()
        atn.modeToStartState.append(atn.states[s])