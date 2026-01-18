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
def canDropLoopEntryEdgeInLeftRecursiveRule(self, config):
    p = config.state
    if p.stateType != ATNState.STAR_LOOP_ENTRY or not p.isPrecedenceDecision or config.context.isEmpty() or config.context.hasEmptyPath():
        return False
    numCtxs = len(config.context)
    for i in range(0, numCtxs):
        returnState = self.atn.states[config.context.getReturnState(i)]
        if returnState.ruleIndex != p.ruleIndex:
            return False
    decisionStartState = p.transitions[0].target
    blockEndStateNum = decisionStartState.endState.stateNumber
    blockEndState = self.atn.states[blockEndStateNum]
    for i in range(0, numCtxs):
        returnStateNumber = config.context.getReturnState(i)
        returnState = self.atn.states[returnStateNumber]
        if len(returnState.transitions) != 1 or not returnState.transitions[0].isEpsilon:
            return False
        returnStateTarget = returnState.transitions[0].target
        if returnState.stateType == ATNState.BLOCK_END and returnStateTarget is p:
            continue
        if returnState is blockEndState:
            continue
        if returnStateTarget is blockEndState:
            continue
        if returnStateTarget.stateType == ATNState.BLOCK_END and len(returnStateTarget.transitions) == 1 and returnStateTarget.transitions[0].isEpsilon and (returnStateTarget.transitions[0].target is p):
            continue
        return False
    return True