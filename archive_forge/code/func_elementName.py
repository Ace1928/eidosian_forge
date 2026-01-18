from io import StringIO
from antlr4.Token import Token
def elementName(self, literalNames: list, symbolicNames: list, a: int):
    if a == Token.EOF:
        return '<EOF>'
    elif a == Token.EPSILON:
        return '<EPSILON>'
    else:
        if a < len(literalNames) and literalNames[a] != '<INVALID>':
            return literalNames[a]
        if a < len(symbolicNames):
            return symbolicNames[a]
        return '<UNKNOWN>'