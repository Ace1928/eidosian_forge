from antlr4.dfa.DFA import DFA
from antlr4.BufferedTokenStream import TokenStream
from antlr4.Lexer import Lexer
from antlr4.Parser import Parser
from antlr4.ParserRuleContext import InterpreterRuleContext, ParserRuleContext
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import StarLoopEntryState, ATNState, LoopEndState
from antlr4.atn.ParserATNSimulator import ParserATNSimulator
from antlr4.PredictionContext import PredictionContextCache
from antlr4.atn.Transition import Transition
from antlr4.error.Errors import RecognitionException, UnsupportedOperationException, FailedPredicateException
def getATNState(self):
    return self.atn.states[self.state]