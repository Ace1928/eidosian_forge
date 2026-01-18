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
def splitAccordingToSemanticValidity(self, configs: ATNConfigSet, outerContext: ParserRuleContext):
    succeeded = ATNConfigSet(configs.fullCtx)
    failed = ATNConfigSet(configs.fullCtx)
    for c in configs:
        if c.semanticContext is not SemanticContext.NONE:
            predicateEvaluationResult = c.semanticContext.eval(self.parser, outerContext)
            if predicateEvaluationResult:
                succeeded.add(c)
            else:
                failed.add(c)
        else:
            succeeded.add(c)
    return (succeeded, failed)