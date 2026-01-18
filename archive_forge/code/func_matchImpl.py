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
def matchImpl(self, tree: ParseTree, patternTree: ParseTree, labels: dict):
    if tree is None:
        raise Exception('tree cannot be null')
    if patternTree is None:
        raise Exception('patternTree cannot be null')
    if isinstance(tree, TerminalNode) and isinstance(patternTree, TerminalNode):
        mismatchedNode = None
        if tree.symbol.type == patternTree.symbol.type:
            if isinstance(patternTree.symbol, TokenTagToken):
                tokenTagToken = patternTree.symbol
                self.map(labels, tokenTagToken.tokenName, tree)
                if tokenTagToken.label is not None:
                    self.map(labels, tokenTagToken.label, tree)
            elif tree.getText() == patternTree.getText():
                pass
            elif mismatchedNode is None:
                mismatchedNode = tree
        elif mismatchedNode is None:
            mismatchedNode = tree
        return mismatchedNode
    if isinstance(tree, ParserRuleContext) and isinstance(patternTree, ParserRuleContext):
        mismatchedNode = None
        ruleTagToken = self.getRuleTagToken(patternTree)
        if ruleTagToken is not None:
            m = None
            if tree.ruleContext.ruleIndex == patternTree.ruleContext.ruleIndex:
                self.map(labels, ruleTagToken.ruleName, tree)
                if ruleTagToken.label is not None:
                    self.map(labels, ruleTagToken.label, tree)
            elif mismatchedNode is None:
                mismatchedNode = tree
            return mismatchedNode
        if tree.getChildCount() != patternTree.getChildCount():
            if mismatchedNode is None:
                mismatchedNode = tree
            return mismatchedNode
        n = tree.getChildCount()
        for i in range(0, n):
            childMatch = self.matchImpl(tree.getChild(i), patternTree.getChild(i), labels)
            if childMatch is not None:
                return childMatch
        return mismatchedNode
    return tree