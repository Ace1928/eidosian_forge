from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def generateRuleBypassTransitions(self, atn: ATN):
    count = len(atn.ruleToStartState)
    atn.ruleToTokenType = [0] * count
    for i in range(0, count):
        atn.ruleToTokenType[i] = atn.maxTokenType + i + 1
    for i in range(0, count):
        self.generateRuleBypassTransition(atn, i)