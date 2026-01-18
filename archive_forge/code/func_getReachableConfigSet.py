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
def getReachableConfigSet(self, input: InputStream, closure: ATNConfigSet, reach: ATNConfigSet, t: int):
    skipAlt = ATN.INVALID_ALT_NUMBER
    for cfg in closure:
        currentAltReachedAcceptState = cfg.alt == skipAlt
        if currentAltReachedAcceptState and cfg.passedThroughNonGreedyDecision:
            continue
        if LexerATNSimulator.debug:
            print('testing', self.getTokenName(t), 'at', str(cfg))
        for trans in cfg.state.transitions:
            target = self.getReachableTarget(trans, t)
            if target is not None:
                lexerActionExecutor = cfg.lexerActionExecutor
                if lexerActionExecutor is not None:
                    lexerActionExecutor = lexerActionExecutor.fixOffsetBeforeMatch(input.index - self.startIndex)
                treatEofAsEpsilon = t == Token.EOF
                config = LexerATNConfig(state=target, lexerActionExecutor=lexerActionExecutor, config=cfg)
                if self.closure(input, config, reach, currentAltReachedAcceptState, True, treatEofAsEpsilon):
                    skipAlt = cfg.alt