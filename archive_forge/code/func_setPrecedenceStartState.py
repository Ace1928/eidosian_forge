from antlr4.atn.ATNState import StarLoopEntryState
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import IllegalStateException
def setPrecedenceStartState(self, precedence: int, startState: DFAState):
    if not self.precedenceDfa:
        raise IllegalStateException('Only precedence DFAs may contain a precedence start state.')
    if precedence < 0:
        return
    if precedence >= len(self.s0.edges):
        ext = [None] * (precedence + 1 - len(self.s0.edges))
        self.s0.edges.extend(ext)
    self.s0.edges[precedence] = startState