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
def closureCheckingStopState(self, config: ATNConfig, configs: ATNConfigSet, closureBusy: set, collectPredicates: bool, fullCtx: bool, depth: int, treatEofAsEpsilon: bool):
    if ParserATNSimulator.debug:
        print('closure(' + str(config) + ')')
    if isinstance(config.state, RuleStopState):
        if not config.context.isEmpty():
            for i in range(0, len(config.context)):
                state = config.context.getReturnState(i)
                if state is PredictionContext.EMPTY_RETURN_STATE:
                    if fullCtx:
                        configs.add(ATNConfig(state=config.state, context=PredictionContext.EMPTY, config=config), self.mergeCache)
                        continue
                    else:
                        if ParserATNSimulator.debug:
                            print('FALLING off rule ' + self.getRuleName(config.state.ruleIndex))
                        self.closure_(config, configs, closureBusy, collectPredicates, fullCtx, depth, treatEofAsEpsilon)
                    continue
                returnState = self.atn.states[state]
                newContext = config.context.getParent(i)
                c = ATNConfig(state=returnState, alt=config.alt, context=newContext, semantic=config.semanticContext)
                c.reachesIntoOuterContext = config.reachesIntoOuterContext
                self.closureCheckingStopState(c, configs, closureBusy, collectPredicates, fullCtx, depth - 1, treatEofAsEpsilon)
            return
        elif fullCtx:
            configs.add(config, self.mergeCache)
            return
        elif ParserATNSimulator.debug:
            print('FALLING off rule ' + self.getRuleName(config.state.ruleIndex))
    self.closure_(config, configs, closureBusy, collectPredicates, fullCtx, depth, treatEofAsEpsilon)