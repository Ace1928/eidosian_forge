from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.PredictionContext import PredictionContext, SingletonPredictionContext, PredictionContextFromRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNState import ATNState, RuleStopState
from antlr4.atn.Transition import WildcardTransition, NotSetTransition, AbstractPredicateTransition, RuleTransition
def getDecisionLookahead(self, s: ATNState):
    if s is None:
        return None
    count = len(s.transitions)
    look = [] * count
    for alt in range(0, count):
        look[alt] = set()
        lookBusy = set()
        seeThruPreds = False
        self._LOOK(s.transition(alt).target, None, PredictionContext.EMPTY, look[alt], lookBusy, set(), seeThruPreds, False)
        if len(look[alt]) == 0 or self.HIT_PRED in look[alt]:
            look[alt] = None
    return look