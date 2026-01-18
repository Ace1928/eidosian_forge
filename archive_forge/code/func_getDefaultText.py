from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def getDefaultText(self):
    return self.getText(self.DEFAULT_PROGRAM_NAME, 0, len(self.tokens.tokens) - 1)