from antlr4.Token import Token
def getSourceInterval(self):
    if self.symbol is None:
        return INVALID_INTERVAL
    tokenIndex = self.symbol.tokenIndex
    return (tokenIndex, tokenIndex)