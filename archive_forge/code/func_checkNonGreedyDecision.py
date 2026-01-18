from io import StringIO
from antlr4.PredictionContext import PredictionContext
from antlr4.atn.ATNState import ATNState, DecisionState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.SemanticContext import SemanticContext
def checkNonGreedyDecision(self, source: LexerATNConfig, target: ATNState):
    return source.passedThroughNonGreedyDecision or (isinstance(target, DecisionState) and target.nonGreedy)