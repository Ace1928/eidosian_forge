from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
def makeLabel(self):
    s = IntervalSet()
    s.addRange(range(self.start, self.stop + 1))
    return s