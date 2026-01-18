from antlr4.PredictionContext import PredictionContextCache, SingletonPredictionContext, PredictionContext
from antlr4.InputStream import InputStream
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import LexerATNConfig
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNConfigSet import ATNConfigSet, OrderedATNConfigSet
from antlr4.atn.ATNState import RuleStopState, ATNState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.Transition import Transition
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import LexerNoViableAltException, UnsupportedOperationException
def getEpsilonTarget(self, input: InputStream, config: LexerATNConfig, t: Transition, configs: ATNConfigSet, speculative: bool, treatEofAsEpsilon: bool):
    c = None
    if t.serializationType == Transition.RULE:
        newContext = SingletonPredictionContext.create(config.context, t.followState.stateNumber)
        c = LexerATNConfig(state=t.target, config=config, context=newContext)
    elif t.serializationType == Transition.PRECEDENCE:
        raise UnsupportedOperationException('Precedence predicates are not supported in lexers.')
    elif t.serializationType == Transition.PREDICATE:
        if LexerATNSimulator.debug:
            print('EVAL rule ' + str(t.ruleIndex) + ':' + str(t.predIndex))
        configs.hasSemanticContext = True
        if self.evaluatePredicate(input, t.ruleIndex, t.predIndex, speculative):
            c = LexerATNConfig(state=t.target, config=config)
    elif t.serializationType == Transition.ACTION:
        if config.context is None or config.context.hasEmptyPath():
            lexerActionExecutor = LexerActionExecutor.append(config.lexerActionExecutor, self.atn.lexerActions[t.actionIndex])
            c = LexerATNConfig(state=t.target, config=config, lexerActionExecutor=lexerActionExecutor)
        else:
            c = LexerATNConfig(state=t.target, config=config)
    elif t.serializationType == Transition.EPSILON:
        c = LexerATNConfig(state=t.target, config=config)
    elif t.serializationType in [Transition.ATOM, Transition.RANGE, Transition.SET]:
        if treatEofAsEpsilon:
            if t.matches(Token.EOF, 0, self.MAX_CHAR_VALUE):
                c = LexerATNConfig(state=t.target, config=config)
    return c