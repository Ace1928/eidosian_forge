from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
def filterForChannel(self, left: int, right: int, channel: int):
    hidden = []
    for i in range(left, right + 1):
        t = self.tokens[i]
        if channel == -1:
            from antlr4.Lexer import Lexer
            if t.channel != Lexer.DEFAULT_TOKEN_CHANNEL:
                hidden.append(t)
        elif t.channel == channel:
            hidden.append(t)
    if len(hidden) == 0:
        return None
    return hidden