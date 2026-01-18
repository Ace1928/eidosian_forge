from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Lexer import Lexer
from antlr4.ListTokenSource import ListTokenSource
from antlr4.Token import Token
from antlr4.error.ErrorStrategy import BailErrorStrategy
from antlr4.error.Errors import RecognitionException, ParseCancellationException
from antlr4.tree.Chunk import TagChunk, TextChunk
from antlr4.tree.RuleTagToken import RuleTagToken
from antlr4.tree.TokenTagToken import TokenTagToken
from antlr4.tree.Tree import ParseTree, TerminalNode, RuleNode
def compileTreePattern(self, pattern: str, patternRuleIndex: int):
    tokenList = self.tokenize(pattern)
    tokenSrc = ListTokenSource(tokenList)
    tokens = CommonTokenStream(tokenSrc)
    from antlr4.ParserInterpreter import ParserInterpreter
    parserInterp = ParserInterpreter(self.parser.grammarFileName, self.parser.tokenNames, self.parser.ruleNames, self.parser.getATNWithBypassAlts(), tokens)
    tree = None
    try:
        parserInterp.setErrorHandler(BailErrorStrategy())
        tree = parserInterp.parse(patternRuleIndex)
    except ParseCancellationException as e:
        raise e.cause
    except RecognitionException as e:
        raise e
    except Exception as e:
        raise CannotInvokeStartRule(e)
    if tokens.LA(1) != Token.EOF:
        raise StartRuleDoesNotConsumeFullPattern()
    from antlr4.tree.ParseTreePattern import ParseTreePattern
    return ParseTreePattern(self, pattern, patternRuleIndex, tree)