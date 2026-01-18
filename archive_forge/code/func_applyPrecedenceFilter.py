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
def applyPrecedenceFilter(self, configs: ATNConfigSet):
    statesFromAlt1 = dict()
    configSet = ATNConfigSet(configs.fullCtx)
    for config in configs:
        if config.alt != 1:
            continue
        updatedContext = config.semanticContext.evalPrecedence(self.parser, self._outerContext)
        if updatedContext is None:
            continue
        statesFromAlt1[config.state.stateNumber] = config.context
        if updatedContext is not config.semanticContext:
            configSet.add(ATNConfig(config=config, semantic=updatedContext), self.mergeCache)
        else:
            configSet.add(config, self.mergeCache)
    for config in configs:
        if config.alt == 1:
            continue
        if not config.precedenceFilterSuppressed:
            context = statesFromAlt1.get(config.state.stateNumber, None)
            if context == config.context:
                continue
        configSet.add(config, self.mergeCache)
    return configSet