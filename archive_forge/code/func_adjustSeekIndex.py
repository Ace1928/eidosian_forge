from antlr4.BufferedTokenStream import BufferedTokenStream
from antlr4.Lexer import Lexer
from antlr4.Token import Token
def adjustSeekIndex(self, i: int):
    return self.nextTokenOnChannel(i, self.channel)