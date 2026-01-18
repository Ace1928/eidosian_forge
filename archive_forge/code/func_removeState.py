from antlr4.IntervalSet import IntervalSet
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import ATNState, DecisionState
def removeState(self, state: ATNState):
    self.states[state.stateNumber] = None