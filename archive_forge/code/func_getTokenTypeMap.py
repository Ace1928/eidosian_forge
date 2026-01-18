from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.error.ErrorListener import ProxyErrorListener, ConsoleErrorListener
def getTokenTypeMap(self):
    tokenNames = self.getTokenNames()
    if tokenNames is None:
        from antlr4.error.Errors import UnsupportedOperationException
        raise UnsupportedOperationException('The current recognizer does not provide a list of token names.')
    result = self.tokenTypeMapCache.get(tokenNames, None)
    if result is None:
        result = zip(tokenNames, range(0, len(tokenNames)))
        result['EOF'] = Token.EOF
        self.tokenTypeMapCache[tokenNames] = result
    return result