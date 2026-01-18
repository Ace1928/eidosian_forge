from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
def getHiddenTokensToLeft(self, tokenIndex: int, channel: int=-1):
    self.lazyInit()
    if tokenIndex < 0 or tokenIndex >= len(self.tokens):
        raise Exception(str(tokenIndex) + ' not in 0..' + str(len(self.tokens) - 1))
    from antlr4.Lexer import Lexer
    prevOnChannel = self.previousTokenOnChannel(tokenIndex - 1, Lexer.DEFAULT_TOKEN_CHANNEL)
    if prevOnChannel == tokenIndex - 1:
        return None
    from_ = prevOnChannel + 1
    to = tokenIndex - 1
    return self.filterForChannel(from_, to, channel)