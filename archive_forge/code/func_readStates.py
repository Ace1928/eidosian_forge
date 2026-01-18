from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def readStates(self, atn: ATN):
    loopBackStateNumbers = []
    endStateNumbers = []
    nstates = self.readInt()
    for i in range(0, nstates):
        stype = self.readInt()
        if stype == ATNState.INVALID_TYPE:
            atn.addState(None)
            continue
        ruleIndex = self.readInt()
        s = self.stateFactory(stype, ruleIndex)
        if stype == ATNState.LOOP_END:
            loopBackStateNumber = self.readInt()
            loopBackStateNumbers.append((s, loopBackStateNumber))
        elif isinstance(s, BlockStartState):
            endStateNumber = self.readInt()
            endStateNumbers.append((s, endStateNumber))
        atn.addState(s)
    for pair in loopBackStateNumbers:
        pair[0].loopBackState = atn.states[pair[1]]
    for pair in endStateNumbers:
        pair[0].endState = atn.states[pair[1]]
    numNonGreedyStates = self.readInt()
    for i in range(0, numNonGreedyStates):
        stateNumber = self.readInt()
        atn.states[stateNumber].nonGreedy = True
    numPrecedenceStates = self.readInt()
    for i in range(0, numPrecedenceStates):
        stateNumber = self.readInt()
        atn.states[stateNumber].isPrecedenceRule = True