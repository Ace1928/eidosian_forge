from antlr4.atn.ATNState import StarLoopEntryState
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import IllegalStateException
def sortedStates(self):
    return sorted(self._states.keys(), key=lambda state: state.stateNumber)