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
def matchPattern(self, tree: ParseTree, pattern: ParseTreePattern):
    labels = dict()
    mismatchedNode = self.matchImpl(tree, pattern.patternTree, labels)
    from antlr4.tree.ParseTreeMatch import ParseTreeMatch
    return ParseTreeMatch(tree, pattern, labels, mismatchedNode)