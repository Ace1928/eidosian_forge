import sys
from antlr4.BufferedTokenStream import TokenStream
from antlr4.CommonTokenFactory import TokenFactory
from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import Token
from antlr4.Lexer import Lexer
from antlr4.atn.ATNDeserializer import ATNDeserializer
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
from antlr4.error.Errors import UnsupportedOperationException, RecognitionException
from antlr4.tree.ParseTreePatternMatcher import ParseTreePatternMatcher
from antlr4.tree.Tree import ParseTreeListener, TerminalNode, ErrorNode
def isExpectedToken(self, symbol: int):
    atn = self._interp.atn
    ctx = self._ctx
    s = atn.states[self.state]
    following = atn.nextTokens(s)
    if symbol in following:
        return True
    if not Token.EPSILON in following:
        return False
    while ctx is not None and ctx.invokingState >= 0 and (Token.EPSILON in following):
        invokingState = atn.states[ctx.invokingState]
        rt = invokingState.transitions[0]
        following = atn.nextTokens(rt.followState)
        if symbol in following:
            return True
        ctx = ctx.parentCtx
    if Token.EPSILON in following and symbol == Token.EOF:
        return True
    else:
        return False