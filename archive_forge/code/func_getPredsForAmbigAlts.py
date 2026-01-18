import sys
from antlr4 import DFA
from antlr4.PredictionContext import PredictionContextCache, PredictionContext, SingletonPredictionContext, \
from antlr4.BufferedTokenStream import TokenStream
from antlr4.Parser import Parser
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNState import StarLoopEntryState, DecisionState, RuleStopState, ATNState
from antlr4.atn.PredictionMode import PredictionMode
from antlr4.atn.SemanticContext import SemanticContext, AND, andContext, orContext
from antlr4.atn.Transition import Transition, RuleTransition, ActionTransition, PrecedencePredicateTransition, \
from antlr4.dfa.DFAState import DFAState, PredPrediction
from antlr4.error.Errors import NoViableAltException
def getPredsForAmbigAlts(self, ambigAlts: set, configs: ATNConfigSet, nalts: int):
    altToPred = [None] * (nalts + 1)
    for c in configs:
        if c.alt in ambigAlts:
            altToPred[c.alt] = orContext(altToPred[c.alt], c.semanticContext)
    nPredAlts = 0
    for i in range(1, nalts + 1):
        if altToPred[i] is None:
            altToPred[i] = SemanticContext.NONE
        elif altToPred[i] is not SemanticContext.NONE:
            nPredAlts += 1
    if nPredAlts == 0:
        altToPred = None
    if ParserATNSimulator.debug:
        print('getPredsForAmbigAlts result ' + str_list(altToPred))
    return altToPred