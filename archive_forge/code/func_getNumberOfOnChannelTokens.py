from antlr4.BufferedTokenStream import BufferedTokenStream
from antlr4.Lexer import Lexer
from antlr4.Token import Token
def getNumberOfOnChannelTokens(self):
    n = 0
    self.fill()
    for i in range(0, len(self.tokens)):
        t = self.tokens[i]
        if t.channel == self.channel:
            n += 1
        if t.type == Token.EOF:
            break
    return n